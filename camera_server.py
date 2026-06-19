#!/usr/bin/env python3
"""
摄像头采集服务器
使用系统 Python 运行，负责从摄像头采集帧并通过网络发送给推理客户端
基于 picamera2 实现，兼容 Raspberry Pi Camera Module 3

性能优化说明：
1. 完整传感器 FOV 模式：1536x864 下采样自 4608x2592，保留广角
2. 零拷贝采集：使用 capture_request() 直接操作缓冲区
3. YUV420 输出：原生格式，数据量减半，无需颜色空间转换
4. 移除 JPEG 编解码：本机 TCP 传输直接发原始数据，省 CPU
"""
import os
import sys
import time
import socket
import struct
import yaml
import traceback
from picamera2 import Picamera2
import numpy as np

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def log_crash(reason: str, exc_info=None):
    """记录崩溃/退出原因到日志文件"""
    try:
        log_path = os.path.join(BASE_DIR, "data", "crash_log.txt")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{timestamp}] 摄像头进程退出 - 原因: {reason}\n")
            if exc_info and exc_info[0] is not None:
                f.write(f"异常类型: {exc_info[0].__name__}\n")
                f.write(f"异常信息: {exc_info[1]}\n")
                f.write("堆栈追踪:\n")
                traceback.print_tb(exc_info[2], file=f)
            f.write(f"{'='*60}\n")
    except Exception:
        pass

# 加载配置
with open(os.path.join(BASE_DIR, "config.yaml"), 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

CAMERA_CFG = CONFIG["camera"]
NETWORK_CFG = CONFIG["network"]

WIDTH = CAMERA_CFG.get("width", 960)
HEIGHT = CAMERA_CFG.get("height", 540)
FPS = CAMERA_CFG.get("fps", 15)
FORMAT = "RGB888"  # picamera2 原生格式，客户端转 BGR
USE_FULL_SENSOR_FOV = CAMERA_CFG.get("use_full_sensor_fov", True)

HOST = NETWORK_CFG.get("host", "127.0.0.1")
PORT = NETWORK_CFG.get("port", 65433)

# YUV420 帧大小计算：W*H*1.5 字节
YUV_FRAME_SIZE = WIDTH * HEIGHT * 3 // 2


def init_camera():
    """初始化摄像头
    优化点：强制使用完整传感器 FOV，避免中心裁剪导致广角丢失
    """
    print(f"[Camera] 初始化摄像头 {WIDTH}x{HEIGHT} @ {FPS}fps, format={FORMAT}")
    if USE_FULL_SENSOR_FOV:
        print(f"[Camera] 强制使用完整传感器阵列（广角最大化）")

    picam2 = Picamera2()

    if USE_FULL_SENSOR_FOV:
        # 关键：指定 sensor 的输出大小，强制 ISP 使用完整传感器阵列下采样
        # Camera Module 3 原生传感器是 4608x2592
        config = picam2.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": FORMAT},
            raw={"size": (4608, 2592)},  # 强制使用完整传感器阵列
            controls={"FrameRate": FPS}
        )
    else:
        # 传统模式（可能中心裁剪，会损失广角）
        config = picam2.create_preview_configuration(
            main={"size": (WIDTH, HEIGHT), "format": FORMAT},
            controls={"FrameRate": FPS}
        )

    picam2.configure(config)
    picam2.start()

    # 预热
    time.sleep(2)
    actual_config = picam2.camera_configuration()
    print(f"[Camera] 实际配置: ")
    print(f"  输出尺寸: {actual_config['main']['size']}")
    print(f"  传感器阵列: {actual_config.get('raw', {}).get('size', 'N/A')}")
    print(f"  像素格式: {actual_config['main']['format']}")
    print("[Camera] 摄像头初始化完成")
    return picam2


def main():
    try:
        # 初始化摄像头
        picam2 = init_camera()
    except Exception as e:
        print(f"[Camera] 摄像头初始化失败: {e}")
        return 1

    # 创建socket服务器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"[Camera] 服务器启动，等待客户端连接: {HOST}:{PORT}")
    except Exception as e:
        print(f"[Camera] 服务器启动失败: {e}")
        picam2.stop()
        return 1

    # 捕获SIGINT
    import signal
    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        running = False
        print("\n[Camera] 收到停止信号")

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    try:
        while running:
            try:
                # 等待客户端连接
                client_socket, addr = server_socket.accept()
                print(f"[Camera] 客户端已连接: {addr}")

                frame_count = 0
                start_time = time.time()

                try:
                    while running:
                        # capture_array() 返回 picamera2 配置的 main.format 格式
                        frame = picam2.capture_array()

                        # 直接发送原始 BGR 数据，跳过 JPEG 编解码
                        frame_data = frame.tobytes()

                        # 发送帧大小和数据
                        client_socket.sendall(struct.pack("<L", len(frame_data)))
                        client_socket.sendall(frame_data)

                        frame_count += 1

                        # 每秒打印一次帧率
                        if frame_count % FPS == 0:
                            elapsed = time.time() - start_time
                            actual_fps = frame_count / elapsed
                            print(f"[Camera] 发送帧率: {actual_fps:.1f}fps", end='\r')

                except (BrokenPipeError, ConnectionResetError):
                    print(f"\n[Camera] 客户端断开连接")
                except Exception as e:
                    print(f"\n[Camera] 传输错误: {e}")
                finally:
                    client_socket.close()
                    print("[Camera] 连接已关闭，等待新的连接...")

            except Exception as e:
                print(f"[Camera] 连接错误: {e}")
                time.sleep(1)

    except KeyboardInterrupt:
        log_crash("用户主动 Ctrl+C 停止")
        raise
    except SystemExit as e:
        log_crash(f"系统主动退出 (code={e.code})")
        raise
    except BaseException as e:
        log_crash("异常崩溃", sys.exc_info())
        raise
    finally:
        print("\n[Camera] 正在停止摄像头...")
        picam2.stop()
        server_socket.close()
        print("[Camera] 服务已退出")

    return 0


if __name__ == "__main__":
    sys.exit(main())
