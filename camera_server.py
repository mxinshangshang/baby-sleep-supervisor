#!/usr/bin/env python3
"""
摄像头采集服务器
使用系统 Python 运行，负责从摄像头采集帧并通过网络发送给推理客户端
基于 picamera2 实现，兼容 Raspberry Pi Camera Module 3
"""
import os
import sys
import time
import socket
import struct
import pickle
import yaml
from picamera2 import Picamera2
import cv2
import numpy as np

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载配置
with open(os.path.join(BASE_DIR, "config.yaml"), 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

CAMERA_CFG = CONFIG["camera"]
NETWORK_CFG = CONFIG["network"]

WIDTH = CAMERA_CFG.get("width", 640)
HEIGHT = CAMERA_CFG.get("height", 480)
FPS = CAMERA_CFG.get("fps", 15)
JPEG_QUALITY = CAMERA_CFG.get("jpeg_quality", 80)

HOST = NETWORK_CFG.get("host", "127.0.0.1")
PORT = NETWORK_CFG.get("port", 65433)


def init_camera():
    """初始化摄像头"""
    print(f"[Camera] 初始化摄像头 {WIDTH}x{HEIGHT} @ {FPS}fps")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "RGB888"},
        controls={"FrameRate": FPS}
    )
    picam2.configure(config)
    picam2.start()

    # 预热
    time.sleep(2)
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

                connection = client_socket.makefile('wb')
                frame_count = 0
                start_time = time.time()

                try:
                    while running:
                        # 采集帧
                        frame = picam2.capture_array()

                        # JPEG压缩
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                        result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)

                        if not result:
                            print("[Camera] 帧编码失败")
                            continue

                        # 序列化
                        data = pickle.dumps(encoded_frame, 0)
                        size = len(data)

                        # 发送帧大小和数据
                        connection.write(struct.pack("<L", size))
                        connection.write(data)
                        connection.flush()

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
                    connection.close()
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
