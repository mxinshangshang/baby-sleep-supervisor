#!/usr/bin/env python3
"""
推理客户端
从摄像头服务器接收帧，进行推理处理，支持预览和无头模式
"""
import sys
import os
import socket
import struct
import pickle
import time
import cv2
import numpy as np
import signal

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.supervision import SleepSupervisor
from src.preview_renderer import PreviewRenderer
from src.notifier import Notifier


def main():
    no_preview = "--no-preview" in sys.argv or "-n" in sys.argv

    # 加载配置
    config = load_config()
    network_cfg = config["network"]
    host = network_cfg.get("host", "127.0.0.1")
    port = network_cfg.get("port", 65433)
    recv_timeout = network_cfg.get("recv_timeout_s", 0.2)

    print("=" * 60)
    print("婴儿睡眠监护系统 - 推理客户端")
    print(f"运行模式: {'后台无头模式' if no_preview else '带预览模式'}")
    print(f"连接地址: {host}:{port}")
    print("=" * 60)

    # 初始化监督器
    try:
        supervisor = SleepSupervisor()
    except Exception as e:
        print(f"监督器初始化失败: {e}")
        return 1

    # 初始化预览渲染器
    renderer = None
    if not no_preview:
        renderer = PreviewRenderer()

    # 初始化通知器
    notifier = Notifier()
    notifier.send_system_notification("婴儿监护系统已启动")

    # 信号处理
    running = True

    def signal_handler(signum, frame):
        nonlocal running
        running = False
        print("\n收到停止信号，正在关闭...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 连接到摄像头服务器
    client_socket = None
    connection = None
    last_connect_attempt = 0
    connect_interval = 5  # 每5秒尝试重连一次

    frame_count = 0
    last_frame_time = time.time()
    fps = 0.0

    try:
        while running:
            # 尝试连接
            if client_socket is None or connection is None:
                now = time.time()
                if now - last_connect_attempt >= connect_interval:
                    print(f"正在连接摄像头服务器 {host}:{port}...")
                    try:
                        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        client_socket.settimeout(5)
                        client_socket.connect((host, port))
                        connection = client_socket.makefile('rb')
                        print("已连接到摄像头服务器")
                    except Exception as e:
                        print(f"连接失败: {e}，{connect_interval}秒后重试")
                        client_socket = None
                        connection = None
                        last_connect_attempt = now

                time.sleep(1)
                continue

            try:
                # 接收帧大小
                size_data = connection.read(struct.calcsize('<L'))
                if not size_data:
                    print("服务器断开连接")
                    break

                size = struct.unpack('<L', size_data)[0]

                # 接收帧数据
                frame_data = connection.read(size)
                if len(frame_data) != size:
                    print("接收帧数据不完整")
                    break

                # 解码帧
                encoded_frame = pickle.loads(frame_data)
                frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)

                if frame is None:
                    print("帧解码失败")
                    continue

                # 处理帧
                results, processed_frame = supervisor.process_frame(frame)

                # 计算FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    now = time.time()
                    fps = 30 / (now - last_frame_time)
                    last_frame_time = now
                    if frame_count % 300 == 0:  # 每300帧打印一次状态
                        status = supervisor.get_status_summary()
                        print(f"运行中... FPS: {fps:.1f} 事件数: {len(status.get('last_events', []))}")

                # 渲染预览
                if renderer is not None:
                    status = supervisor.get_status_summary()
                    render_frame = renderer.render(
                        processed_frame,
                        results,
                        status,
                        supervisor.region_detector
                    )

                    key = renderer.show(render_frame)

                    # 处理快捷键
                    if key == ord('q'):
                        # 发送信号给主进程，由主进程决定重启还是退出
                        os.kill(os.getppid(), signal.SIGUSR1)
                    elif key == ord('h'):
                        renderer.show_help = not renderer.show_help
                    elif key == ord('d'):
                        renderer.show_detection_boxes = not renderer.show_detection_boxes
                    elif key == ord('r'):
                        renderer.show_safe_region = not renderer.show_safe_region
                    elif key == ord('s'):
                        renderer.show_statistics = not renderer.show_statistics
                    elif key == ord('c'):
                        print("区域校准功能请运行 calibrate_region.py")

            except (socket.timeout, ConnectionResetError, BrokenPipeError) as e:
                print(f"连接错误: {e}")
                break
            except Exception as e:
                print(f"处理帧出错: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
                continue

    finally:
        print("正在关闭资源...")

        # 关闭连接
        if connection:
            connection.close()
        if client_socket:
            client_socket.close()

        # 关闭监督器
        supervisor.close()

        # 关闭渲染器
        if renderer:
            renderer.close()

        notifier.send_system_notification("婴儿监护系统已停止")
        print("系统已正常退出")

    return 0


if __name__ == "__main__":
    sys.exit(main())
