#!/usr/bin/env python3
"""
推理客户端
从摄像头服务器接收帧，进行推理处理，支持预览和无头模式。

关键设计：摄像头接收/预览/算法检测三者解耦。
- 接收线程持续读取摄像头帧，始终保留最新帧，避免 socket 反压拖慢 camera_server。
- 主线程按 preview.display_fps 刷新预览。
- 算法按 inference.inference_fps 抽取最新帧检测；高温时只降低算法检测帧率，不降低预览帧率。
"""
import sys
import os
import socket
import struct
import pickle
import time
import cv2
import signal
import threading
from typing import Optional

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.supervision import SleepSupervisor
from src.preview_renderer import PreviewRenderer
from src.notifier import Notifier


def read_cpu_temp_c():
    """Best-effort Raspberry Pi CPU temperature reader."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r', encoding='utf-8') as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return None


class LatestFrameReceiver:
    """Continuously drains the camera TCP stream and keeps only the latest decoded frame."""

    def __init__(self, connection, frame_width: int, frame_height: int):
        self.connection = connection
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_seq = 0
        self.latest_time = 0.0
        self.running = True
        self.error: Optional[BaseException] = None
        self.thread = threading.Thread(target=self._run, name="camera-frame-receiver", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False

    def snapshot(self):
        with self.lock:
            if self.latest_frame is None:
                return None, self.latest_seq, self.latest_time
            return self.latest_frame.copy(), self.latest_seq, self.latest_time

    def _run(self):
        header_size = struct.calcsize('<L')
        try:
            while self.running:
                size_data = self.connection.read(header_size)
                if not size_data:
                    raise ConnectionError("服务器断开连接")
                size = struct.unpack('<L', size_data)[0]
                frame_data = self.connection.read(size)
                if len(frame_data) != size:
                    raise ConnectionError("接收帧数据不完整")

                # 优化点：直接接收原始 YUV420 数据，跳过 JPEG 解码
                # YUV420 数据结构：Y(W*H) + U(W/2*H/2) + V(W/2*H/2)
                frame_yuv = np.frombuffer(frame_data, dtype=np.uint8)

                # 重构 YUV420 数组形状 (H*3//2, W)
                frame_yuv_reshaped = frame_yuv.reshape((self.frame_height * 3 // 2, self.frame_width))

                # 转换为 BGR（只做一次，用于预览和检测）
                frame_bgr = cv2.cvtColor(frame_yuv_reshaped, cv2.COLOR_YUV2BGR_I420)

                with self.lock:
                    self.latest_frame = frame_bgr
                    self.latest_seq += 1
                    self.latest_time = time.time()
        except BaseException as e:
            self.error = e
            self.running = False



class LatestInferenceWorker:
    """Runs heavy supervision on latest frames at target inference FPS.

    The preview thread never waits for this worker. It renders the latest camera frame
    with the most recent completed detection result.
    """

    def __init__(self, supervisor: SleepSupervisor, receiver: LatestFrameReceiver,
                 normal_fps: float, throttle_fps: float, temp_warn_c: float,
                 thermal_enabled: bool, temp_check_interval: float):
        self.supervisor = supervisor
        self.receiver = receiver
        self.normal_fps = max(1.0, normal_fps)
        self.throttle_fps = max(1.0, throttle_fps)
        self.target_fps = self.normal_fps
        self.external_max_fps = self.normal_fps
        self.temp_warn_c = temp_warn_c
        self.thermal_enabled = thermal_enabled
        self.temp_check_interval = temp_check_interval
        self.last_temp_check = 0.0
        self.last_temp_c = None
        self.last_inferred_seq = -1
        self.latest_results = {"timestamp": time.time(), "fps": 0.0, "events": [], "detections": {}}
        self.latest_results_seq = -1
        self.latest_results_time = 0.0
        self.count = 0
        self.actual_fps = 0.0
        self._last_stats_time = time.time()
        self.lock = threading.Lock()
        self.running = True
        self.error: Optional[BaseException] = None
        self.thread = threading.Thread(target=self._run, name="baby-inference-worker", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False

    def set_external_max_fps(self, fps: float):
        self.external_max_fps = max(1.0, float(fps))

    def snapshot(self):
        with self.lock:
            return dict(self.latest_results), self.latest_results_seq, self.latest_results_time, self.actual_fps, self.target_fps, self.last_temp_c

    def _run(self):
        next_inference_time = 0.0
        try:
            while self.running:
                now = time.time()
                if self.thermal_enabled and now - self.last_temp_check >= self.temp_check_interval:
                    self.last_temp_check = now
                    self.last_temp_c = read_cpu_temp_c()
                    base_fps = self.throttle_fps if (self.last_temp_c is not None and self.last_temp_c >= self.temp_warn_c) else self.normal_fps
                    self.target_fps = max(1.0, min(base_fps, self.external_max_fps))

                if now < next_inference_time:
                    time.sleep(min(0.02, next_inference_time - now))
                    continue

                frame, seq, _ = self.receiver.snapshot()
                if frame is None or seq == self.last_inferred_seq:
                    time.sleep(0.005)
                    continue

                results, _ = self.supervisor.process_frame(frame)
                self.last_inferred_seq = seq
                self.count += 1
                with self.lock:
                    self.latest_results = results
                    self.latest_results_seq = seq
                    self.latest_results_time = time.time()

                now = time.time()
                inference_interval = 1.0 / max(1.0, self.target_fps)
                if next_inference_time <= 0:
                    next_inference_time = now + inference_interval
                else:
                    next_inference_time += inference_interval
                    if next_inference_time < now - inference_interval:
                        next_inference_time = now + inference_interval
                if now - self._last_stats_time >= 30.0:
                    elapsed = now - self._last_stats_time
                    self.actual_fps = self.count / max(1e-6, elapsed)
                    self.count = 0
                    self._last_stats_time = now
        except BaseException as e:
            self.error = e
            self.running = False


def main():
    no_preview = "--no-preview" in sys.argv or "-n" in sys.argv

    config = load_config()
    network_cfg = config["network"]
    host = network_cfg.get("host", "127.0.0.1")
    port = network_cfg.get("port", 65433)

    camera_cfg = config.get("camera", {})
    frame_width = camera_cfg.get("width", 1536)
    frame_height = camera_cfg.get("height", 864)

    inference_cfg = config.get("inference", {})
    preview_cfg = config.get("preview", {})
    thermal_cfg = config.get("thermal", {})

    normal_inference_fps = max(1.0, float(inference_cfg.get("inference_fps", 8)))
    target_inference_fps = normal_inference_fps
    preview_fps_target = max(1.0, float(preview_cfg.get("display_fps", 10)))
    throttle_inference_fps = max(1.0, float(thermal_cfg.get("throttle_inference_fps", 5)))
    temp_warn_c = float(thermal_cfg.get("temp_warn_c", 65.0))
    thermal_enabled = bool(thermal_cfg.get("enabled", True))
    temp_check_interval = float(thermal_cfg.get("temp_check_interval_s", 10.0))
    last_temp_c = None

    print("=" * 60)
    print("婴儿睡眠监护系统 - 推理客户端")
    print(f"运行模式: {'后台无头模式' if no_preview else '带预览模式'}")
    print(f"连接地址: {host}:{port}")
    print(f"预览目标FPS: {preview_fps_target:.1f} | 算法目标FPS: {target_inference_fps:.1f}")
    print("=" * 60)

    try:
        supervisor = SleepSupervisor()
    except Exception as e:
        print(f"监督器初始化失败: {e}")
        return 1

    renderer = None if no_preview else PreviewRenderer()
    notifier = Notifier()
    notifier.send_system_notification("婴儿监护系统已启动")

    running = True

    def signal_handler(signum, frame):
        nonlocal running
        running = False
        print("\n收到停止信号，正在关闭...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    client_socket = None
    connection = None
    receiver: Optional[LatestFrameReceiver] = None
    last_connect_attempt = 0.0
    connect_interval = 5.0

    inference_worker: Optional[LatestInferenceWorker] = None
    next_preview_time = 0.0

    preview_count = 0
    last_stats_time = time.time()
    preview_fps_actual = 0.0
    adaptive_infer_max_fps = normal_inference_fps

    try:
        while running:
            if receiver is None or not receiver.running:
                if receiver and receiver.error:
                    print(f"摄像头接收线程退出: {receiver.error}")
                if inference_worker:
                    inference_worker.stop()
                    inference_worker = None
                if connection:
                    try:
                        connection.close()
                    except Exception:
                        pass
                if client_socket:
                    try:
                        client_socket.close()
                    except Exception:
                        pass
                connection = None
                client_socket = None
                receiver = None

                now = time.time()
                if now - last_connect_attempt >= connect_interval:
                    last_connect_attempt = now
                    print(f"正在连接摄像头服务器 {host}:{port}...")
                    try:
                        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        client_socket.settimeout(5)
                        client_socket.connect((host, port))
                        client_socket.settimeout(None)
                        connection = client_socket.makefile('rb')
                        receiver = LatestFrameReceiver(connection, frame_width, frame_height)
                        receiver.start()
                        inference_worker = LatestInferenceWorker(
                            supervisor, receiver, normal_inference_fps, throttle_inference_fps,
                            temp_warn_c, thermal_enabled, temp_check_interval
                        )
                        inference_worker.start()
                        print("已连接到摄像头服务器，接收线程和算法线程已启动")
                    except Exception as e:
                        print(f"连接失败: {e}，{connect_interval:.0f}秒后重试")
                        try:
                            if connection:
                                connection.close()
                            if client_socket:
                                client_socket.close()
                        except Exception:
                            pass
                        connection = None
                        client_socket = None
                        receiver = None
                time.sleep(0.1)
                continue

            frame, seq, frame_time = receiver.snapshot()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            results, result_seq, result_time, inference_fps_actual, target_inference_fps, last_temp_c = (
                inference_worker.snapshot() if inference_worker else ({"timestamp": now, "fps": 0.0, "events": [], "detections": {}}, -1, 0.0, 0.0, normal_inference_fps, None)
            )

            # 预览：按 display_fps 刷新，用最新帧 + 算法线程最近一次检测结果。
            should_preview = no_preview or now >= next_preview_time
            if should_preview:
                if renderer is not None:
                    status = dict(supervisor.get_status_summary())
                    status["fps"] = preview_fps_actual
                    render_results = dict(results)
                    render_results["fps"] = preview_fps_actual
                    render_frame = renderer.render(
                        frame,
                        render_results,
                        status,
                        supervisor.region_detector
                    )
                    key = renderer.show(render_frame)
                    if key == ord('q'):
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
                    preview_count += 1
                preview_interval = 1.0 / preview_fps_target
                if next_preview_time <= 0:
                    next_preview_time = now + preview_interval
                else:
                    # Fixed-rate scheduling: do not add render time to the frame interval.
                    next_preview_time += preview_interval
                    if next_preview_time < now - preview_interval:
                        next_preview_time = now + preview_interval

            if now - last_stats_time >= 30.0:
                elapsed = now - last_stats_time
                preview_fps_actual = preview_count / elapsed if renderer is not None else 0.0
                preview_count = 0
                last_stats_time = now
                if inference_worker is not None:
                    # Preview has priority. If UI cannot keep up, reduce algorithm FPS; if it recovers, step back up.
                    if preview_fps_actual < preview_fps_target * 0.80:
                        adaptive_infer_max_fps = max(2.0, adaptive_infer_max_fps - 1.0)
                    elif preview_fps_actual >= preview_fps_target * 0.92 and adaptive_infer_max_fps < normal_inference_fps:
                        adaptive_infer_max_fps = min(normal_inference_fps, adaptive_infer_max_fps + 0.5)
                    inference_worker.set_external_max_fps(adaptive_infer_max_fps)
                temp_text = f" temp={last_temp_c:.1f}C" if last_temp_c is not None else ""
                print(
                    f"运行中... preview_fps={preview_fps_actual:.1f} "
                    f"infer_fps={inference_fps_actual:.1f} target_infer={target_inference_fps:.1f} "
                    f"infer_cap={adaptive_infer_max_fps:.1f} camera_seq={seq} result_seq={result_seq}{temp_text}",
                    flush=True,
                )

            time.sleep(0.001)

    finally:
        print("正在关闭资源...")
        if inference_worker:
            inference_worker.stop()
        if receiver:
            receiver.stop()
        if connection:
            try:
                connection.close()
            except Exception:
                pass
        if client_socket:
            try:
                client_socket.close()
            except Exception:
                pass
        supervisor.close()
        if renderer:
            renderer.close()
        notifier.send_system_notification("婴儿监护系统已停止")
        print("系统已正常退出")

    return 0


if __name__ == "__main__":
    sys.exit(main())
