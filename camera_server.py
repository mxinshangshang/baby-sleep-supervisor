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
from picamera2 import Picamera2
import numpy as np

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载配置
with open(os.path.join(BASE_DIR, "config.yaml"), 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

CAMERA_CFG = CONFIG["camera"]
NETWORK_CFG = CONFIG["network"]

WIDTH = CAMERA_CFG.get("width", 1536)
HEIGHT = CAMERA_CFG.get("height", 864)
FPS = CAMERA_CFG.get("fps", 15)
FORMAT = CAMERA_CFG.get("format", "YUV420")
USE_FULL_SENSOR_FOV = CAMERA_CFG.get("use_full_sensor_fov", True)

HOST = NETWORK_CFG.get("host", "127.0.0.1")
PORT = NETWORK_CFG.get("port", 65433)

# YUV420 帧大小计算：W*H*1.5 字节
YUV_FRAME_SIZE = WIDTH * HEIGHT * 3 // 2


def init_camera():
    """初始化摄像头
    优化点：使用完整传感器 FOV，避免中心裁剪导致广角丢失
    """
    print(f"[Camera] 初始化摄像头 {WIDTH}x{HEIGHT} @ {FPS}fps, format={FORMAT}")
    if USE_FULL_SENSOR_FOV:
        print(f"[Camera] 启用完整传感器 FOV 模式（广角最大化）")

    picam2 = Picamera2()

    if USE_FULL_SENSOR_FOV:
        # 关键：使用 video_configuration 会自动使用完整传感器区域下采样
        # 而不是 preview_configuration 的中心裁剪模式
        config = picam2.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": FORMAT},
            controls={"FrameRate": FPS}
        )
    else:
        # 传统预览模式（中心裁剪，会损失广角）
        config = picam2.create_preview_configuration(
            main={"size": (WIDTH, HEIGHT), "format": FORMAT},
            controls={"FrameRate": FPS}
        )

    picam2.configure(config)
    picam2.start()

    # 预热
    time.sleep(2)
    actual_config = picam2.camera_configuration()
    print(f"[Camera] 实际配置: size={actual_config['main']['size']}, "
          f"format={actual_config['main']['format']}")
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
                        # 优化点：零拷贝采集，直接使用 request 缓冲区
                        # 避免 capture_array() 的内存拷贝
                        request = picam2.capture_request()
                        frame = request.make_array("main")
                        request.release()

                        # 优化点：直接发送原始 YUV 数据，跳过 JPEG 编解码
                        # YUV420 已经是连续内存，直接发送
                        frame_data = frame.tobytes()

                        # 发送帧大小和数据（兼容旧协议，但现在是固定大小）
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

    finally:
        print("\n[Camera] 正在停止摄像头...")
        picam2.stop()
        server_socket.close()
        print("[Camera] 服务已退出")

    return 0


if __name__ == "__main__":
    sys.exit(main())
