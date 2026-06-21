#!/usr/bin/env python3
"""
摄像头采集服务器
使用系统 Python 运行，负责从摄像头采集帧并通过网络发送给推理客户端
基于 picamera2 实现，兼容 Raspberry Pi Camera Module 3
性能优化说明：
1. 完整传感器 FOV 模式：960x540 下采样自 4608x2592，保留广角
2. 零拷贝采集：使用 capture_request() 直接操作缓冲区
3. RGB888 输出：原生格式，无需颜色空间转换
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
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 加载配置
with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

CAMERA_CFG = CONFIG["camera"]
NETWORK_CFG = CONFIG["network"]
DUAL_CFG = CONFIG.get("dual_camera", {}) or {}
DUAL_ENABLED = bool(DUAL_CFG.get("enabled", False))

WIDTH = CAMERA_CFG.get("width", 960)
HEIGHT = CAMERA_CFG.get("height", 540)
FPS = CAMERA_CFG.get("fps", 15)
FORMAT = "RGB888"
USE_FULL_SENSOR_FOV = CAMERA_CFG.get("use_full_sensor_fov", True)

HOST = NETWORK_CFG.get("host", "127.0.0.1")
PORT = NETWORK_CFG.get("port", 65433)

def _build_picam_config(picam):
    if USE_FULL_SENSOR_FOV:
        return picam.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": FORMAT},
            raw={"size": (4608, 2592)},
            controls={"FrameRate": FPS},
        )
    return picam.create_video_configuration(
        main={"size": (WIDTH, HEIGHT), "format": FORMAT},
        controls={"FrameRate": FPS},
    )

def init_camera():
    print(f"[Camera] 初始化摄像头 {WIDTH}x{HEIGHT} @ {FPS}fps, format={FORMAT}")
    if USE_FULL_SENSOR_FOV:
        print(f"[Camera] 强制使用完整传感器阵列（广角最大化）")

    if DUAL_ENABLED:
        try:
            from src.camera.dual_camera_proxy import CameraRouter
            router = CameraRouter(
                config_builder=_build_picam_config,
                dual_cfg=DUAL_CFG,
                storage=None,
            )
            router.start()
            actual_config = router.camera_configuration()
            print(f"[Camera] 实际配置（router透传active摄像头）:")
            print(f"  输出尺寸: {actual_config.get('main', {}).get('size', 'N/A')}")
            print(f"  传感器阵列: {actual_config.get('raw', {}).get('size', 'N/A')}")
            print(f"  像素格式: {actual_config.get('main', {}).get('format', 'N/A')}")
            print(f"  active_camera_id: {router.get_status().get('active_camera_id')}")
            print("[Camera] CameraRouter初始化完成（双摄模式）")
            return router
        except Exception as e:
            print(f"[Camera][WARN] 双摄路由初始化失败，降级为单摄模式: {e}")
            traceback.print_exc()

    cam = Picamera2()
    config = _build_picam_config(cam)
    cam.configure(config)
    cam.start()
    time.sleep(2)
    actual_config = cam.camera_configuration()
    print(f"[Camera] 实际配置: ")
    print(f"  输出尺寸: {actual_config['main']['size']}")
    print(f"  传感器阵列: {actual_config.get('raw', {}).get('size', 'N/A')}")
    print(f"  像素格式: {actual_config['main']['format']}")
    print("[Camera] 摄像头初始化完成（单摄模式）")
    return cam

def main():
    try:
        cam = init_camera()
    except Exception as e:
        print(f"[Camera] 初始化失败: {e}")
        traceback.print_exc()
        return 1

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"[Camera] 服务器启动，等待客户端连接: {HOST}:{PORT}")
    except Exception as e:
        print(f"[Camera] 服务器启动失败: {e}")
        cam.stop()
        return 1

    import signal
    running = True

    def handle_sigint(signum, frame):
        nonlocal running
        running = False
        print("\n[Camera] 收到停止信号")

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    SWITCH_CMD_PATH = os.path.join(BASE_DIR, "data", "dual_camera_switch.cmd")

    def handle_force_switch(signum, frame):
        if not hasattr(cam, "force_switch"):
            print("[Camera] 收到信号但非双摄模式，忽略")
            return
        try:
            with open(SWITCH_CMD_PATH, "r", encoding="utf-8") as f:
                target = int(f.read().strip())
            import threading
            def _do():
                try:
                    ok = cam.force_switch(target)
                    print(f"[Camera] 强制切换结果: {ok}")
                except Exception as e:
                    print(f"[Camera] 强制切换失败: {e}")
            threading.Thread(target=_do, daemon=True).start()
        except Exception as e:
            print(f"[Camera] 信号处理失败: {e}")

    signal.signal(signal.SIGUSR2, handle_force_switch)

    try:
        client_socket, addr = server_socket.accept()
        print(f"[Camera] 接受客户端连接: {addr}")
        client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        frame_count = 0
        while running:
            try:
                frame = cam.capture_array()
                frame_bytes = frame.tobytes()
                header = struct.pack("<L", len(frame_bytes))
                client_socket.sendall(header + frame_bytes)
                frame_count += 1
                if frame_count % 150 == 0:
                    print(f"[Camera] 已发送 {frame_count} 帧")
            except Exception as e:
                print(f"[Camera] 发送失败: {e}")
                break
    finally:
        print("[Camera] 正在停止")
        cam.stop()
        server_socket.close()
        if "client_socket" in locals():
            client_socket.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
