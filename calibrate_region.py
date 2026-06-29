#!/usr/bin/env python3
"""
安全区域校准工具
允许用户通过鼠标框选设置婴儿床的安全区域
"""
import sys
import os

# 自动使用kid项目虚拟环境（和main.py架构一致）
VENV_PYTHON = "/home/mxin/.openclaw/workspace/kid_supervisor_v3/venv_311/bin/python"
if sys.executable != VENV_PYTHON:
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

import socket
import struct
import pickle
import subprocess
import time
import cv2
import numpy as np
import yaml

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import CONFIG_PATH


ALERT_OPTIONS = [
    ("cry_detected", "Crying"),
    ("occlusion_detected", "Face covered"),
    ("limb_exposure", "Limb exposed"),
    ("region_exit", "Out of safe region"),
    ("region_enter", "Enter safe region"),
    ("prone_detected", "Prone sleeping risk"),
    ("face_not_visible", "Face not visible"),
]

# 告警类型到检测开关的映射
ALERT_TO_DETECTION_FLAGS = {
    "cry_detected": ["cry_detection_enabled"],
    "occlusion_detected": ["occlusion_detection_enabled"],
    "limb_exposure": ["limb_exposure_enabled"],
    "region_exit": ["region_detection_enabled"],
    "region_enter": ["region_detection_enabled"],
    "prone_detected": ["prone_detection_enabled"],
    "face_not_visible": ["face_absence_detection_enabled"],
}


class RegionCalibrator:
    def __init__(self):
        self.window_name = "Baby Region Calibrator"
        self.points = []
        self.current_region = None
        self.frame = None
        self.temp_frame = None
        self.is_completed = False

        # 加载现有配置，但校准启动时不显示旧区域，直接等待新四点输入
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        existing_region = self.config["detection"].get("safe_region", [[50, 50], [590, 430]])
        print(f"Loaded existing safe_region with {len(existing_region)} points. Click 4 new corners to replace it.")

        self.client_socket = None
        self.connection = None
        self.camera_proc = None
        network_cfg = self.config.get("network", {})
        self.host = network_cfg.get("host", "127.0.0.1")
        self.port = network_cfg.get("port", 65433)
        self.width = self.config["camera"].get("width", 640)
        self.height = self.config["camera"].get("height", 480)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调"""
        # 限制坐标在画面范围内
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_completed or len(self.points) >= 4:
                self.points = []
                self.current_region = None
                self.is_completed = False
                print("Start a new 4-corner region")

            self.points.append((x, y))
            print(f"Point #{len(self.points)}: ({x}, {y})")

            if len(self.points) == 4:
                self.is_completed = True
                self.current_region = self.points.copy()
                print(f"Region auto completed with 4 corners: {self.current_region}")

            self._draw_polygon()

        elif event == cv2.EVENT_LBUTTONDBLCLK:
            pass

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键撤销上一个顶点
            if self.points and not self.is_completed:
                removed = self.points.pop()
                print(f"撤销顶点: {removed}, 剩余{len(self.points)}个")
                self._draw_polygon()

    def _draw_polygon(self):
        """在当前帧上绘制多边形"""
        if not self.frame is None:
            self.temp_frame = self.frame.copy()
            if len(self.points) >= 1:
                # 绘制所有顶点
                for i, p in enumerate(self.points):
                    cv2.circle(self.temp_frame, p, 4, (0, 255, 0), -1)
                    cv2.putText(self.temp_frame, str(i+1), (p[0]+5, p[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # 绘制边
                for i in range(len(self.points) - 1):
                    cv2.line(self.temp_frame, self.points[i], self.points[i+1], (0, 255, 0), 2)

                # 如果完成了，绘制闭合边和填充
                if self.is_completed and len(self.points) >= 3:
                    cv2.line(self.temp_frame, self.points[-1], self.points[0], (0, 255, 0), 2)
                    # 半透明填充
                    overlay = self.temp_frame.copy()
                    cv2.fillPoly(overlay, [np.array(self.points, np.int32)], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.2, self.temp_frame, 0.8, 0, self.temp_frame)
                    cv2.putText(self.temp_frame, "Region locked, press s to save", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif len(self.points) >= 2:
                    # 未完成，绘制到鼠标当前位置的临时线
                    pass

            cv2.imshow(self.window_name, self.temp_frame)

    def init_camera(self):
        """连接摄像头服务器，必要时自动启动"""
        if self.connect_camera_server(timeout=1):
            return True

        print("未发现运行中的摄像头服务器，正在自动启动 camera_server.py...")
        try:
            camera_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_server.py")
            self.camera_proc = subprocess.Popen(["/usr/bin/python3", camera_script])
        except Exception as e:
            print(f"启动 camera_server.py 失败: {e}")
            return False

        deadline = time.time() + 10
        while time.time() < deadline:
            if self.camera_proc.poll() is not None:
                print(f"camera_server.py 已退出，返回码: {self.camera_proc.returncode}")
                return False
            if self.connect_camera_server(timeout=1):
                return True
            time.sleep(0.5)

        print("等待 camera_server.py 启动超时")
        self.close_camera()
        return False

    def connect_camera_server(self, timeout=5):
        print(f"正在连接摄像头服务器 {self.host}:{self.port}...")
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(timeout)
            self.client_socket.connect((self.host, self.port))
            self.connection = self.client_socket.makefile('rb')
            print("摄像头服务器连接完成")
            return True
        except Exception:
            self.close_connection()
            return False

    def read_frame(self):
        header_size = struct.calcsize('<LL')
        header_data = self.connection.read(header_size)
        if not header_data:
            raise RuntimeError("摄像头服务器断开连接")

        size, camera_id = struct.unpack('<LL', header_data)
        frame_data = self.connection.read(size)
        if len(frame_data) != size:
            raise RuntimeError("接收帧数据不完整")

        w = self.config["camera"]["width"]
        h = self.config["camera"]["height"]

        # 新格式：原始 BGR 数据（无对齐问题）
        expected_bgr_size = w * h * 3
        if len(frame_data) == expected_bgr_size:
            frame_bgr = np.frombuffer(frame_data, dtype=np.uint8)
            return frame_bgr.reshape((h, w, 3))

        # 回退到旧的 pickle+JPEG 格式
        try:
            encoded_frame = pickle.loads(frame_data)
            frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("帧解码失败")
            return frame
        except Exception as e:
            raise RuntimeError(f"帧格式不兼容: {e}")

    def close_connection(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.client_socket is not None:
            self.client_socket.close()
            self.client_socket = None

    def close_camera(self):
        self.close_connection()
        if self.camera_proc is not None and self.camera_proc.poll() is None:
            self.camera_proc.terminate()
            try:
                self.camera_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.camera_proc.kill()
                self.camera_proc.wait()
        self.camera_proc = None

    def save_config(self):
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False

    def save_region(self):
        """保存区域到配置文件"""
        if not self.current_region or len(self.current_region) < 4:
            print("Please select 4 corners before saving")
            return False

        self.config["detection"]["safe_region"] = [[int(p[0]), int(p[1])] for p in self.current_region]

        if self.save_config():
            print(f"Safe region saved to: {CONFIG_PATH}")
            print(f"New region: {len(self.current_region)} corners")
            return True
        return False

    def notification_mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        selected = param

        mid = len(ALERT_OPTIONS) // 2 + 1
        col1 = ALERT_OPTIONS[:mid]
        col2 = ALERT_OPTIONS[mid:]

        # 检查第一列
        for index, (event_type, _) in enumerate(col1):
            y_pos = 140 + index * 50
            if 40 <= x <= 300 and (y_pos - 25) <= y <= (y_pos + 15):
                if event_type in selected:
                    selected.remove(event_type)
                else:
                    selected.add(event_type)
                return

        # 检查第二列
        for offset, (event_type, _) in enumerate(col2):
            y_pos = 140 + offset * 50
            if 320 <= x <= 600 and (y_pos - 25) <= y <= (y_pos + 15):
                if event_type in selected:
                    selected.remove(event_type)
                else:
                    selected.add(event_type)
                return

    def draw_notification_options(self, selected):
        frame = np.full((460, 640, 3), 248, dtype=np.uint8)

        # 标题
        cv2.putText(frame, "Select Feishu Alert Types", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
        cv2.putText(frame, "What you select = What will be detected & notified", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # 分隔线
        cv2.line(frame, (30, 95), (610, 95), (200, 200, 200), 1)

        # 两列布局
        mid = len(ALERT_OPTIONS) // 2 + 1
        col1 = ALERT_OPTIONS[:mid]
        col2 = ALERT_OPTIONS[mid:]

        # 第一列
        for index, (event_type, label) in enumerate(col1):
            y = 140 + index * 50
            checked = event_type in selected

            # 复选框
            cv2.rectangle(frame, (50, y - 20), (80, y + 10), (60, 60, 60), 2)
            if checked:
                cv2.line(frame, (57, y - 10), (66, y), (0, 150, 0), 3)
                cv2.line(frame, (66, y), (78, y - 16), (0, 150, 0), 3)

            # 数字和标签
            cv2.putText(frame, f"{index + 1}.", (95, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
            cv2.putText(frame, label, (130, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)

        # 第二列
        for offset, (event_type, label) in enumerate(col2):
            index = mid + offset
            y = 140 + offset * 50
            checked = event_type in selected

            # 复选框
            cv2.rectangle(frame, (330, y - 20), (360, y + 10), (60, 60, 60), 2)
            if checked:
                cv2.line(frame, (337, y - 10), (346, y), (0, 150, 0), 3)
                cv2.line(frame, (346, y), (358, y - 16), (0, 150, 0), 3)

            # 数字和标签
            cv2.putText(frame, f"{index + 1}.", (375, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
            cv2.putText(frame, label, (410, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)

        # 底部分隔线
        cv2.line(frame, (30, 390), (610, 390), (200, 200, 200), 1)

        # 底部操作提示
        cv2.putText(frame, "Keyboard shortcuts:", (40, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        cv2.putText(frame, "1-7: toggle   a: all   n: none   Enter/s: save   q: skip", (40, 445),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)

        return frame

    def _sync_detection_flags(self, enabled_alerts):
        """根据选择的告警类型自动同步检测开关"""
        detect_cfg = self.config.setdefault("detection", {})

        # 首先收集所有需要开启的检测项
        flags_to_enable = set()
        for alert_type in enabled_alerts:
            flags_to_enable.update(ALERT_TO_DETECTION_FLAGS.get(alert_type, []))

        # 根据是否需要来设置开关
        # 1. Cry detection
        detect_cfg["cry_detection_enabled"] = "cry_detection_enabled" in flags_to_enable
        # 2. Occlusion detection
        detect_cfg["occlusion_detection_enabled"] = "occlusion_detection_enabled" in flags_to_enable
        # 3. Limb exposure detection
        detect_cfg["limb_exposure_enabled"] = "limb_exposure_enabled" in flags_to_enable
        # 4. Region detection (both exit and enter need this)
        detect_cfg["region_detection_enabled"] = "region_detection_enabled" in flags_to_enable
        # 5. Prone detection
        detect_cfg["prone_detection_enabled"] = "prone_detection_enabled" in flags_to_enable
        # 6. Face absence detection
        detect_cfg["face_absence_detection_enabled"] = "face_absence_detection_enabled" in flags_to_enable

        print(f"Synced detection flags:")
        for flag in ["cry_detection_enabled", "occlusion_detection_enabled",
                     "limb_exposure_enabled", "region_detection_enabled",
                     "prone_detection_enabled", "face_absence_detection_enabled"]:
            status = "ON" if detect_cfg.get(flag, False) else "OFF"
            print(f"  {flag:30s}: {status}")

    def configure_notification_alerts(self):
        notify_cfg = self.config.setdefault("notification", {})
        default_enabled = [event_type for event_type, _ in ALERT_OPTIONS]
        enabled = notify_cfg.get("enabled_alert_types", default_enabled)
        selected = {event_type for event_type in enabled if event_type in default_enabled}

        cv2.namedWindow("Notification Alerts", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Notification Alerts", self.notification_mouse_callback, selected)

        while True:
            cv2.imshow("Notification Alerts", self.draw_notification_options(selected))
            key = cv2.waitKey(50) & 0xFF

            if key in (13, 10, ord('s')):
                # 保存通知设置
                notify_cfg["enabled_alert_types"] = [event_type for event_type, _ in ALERT_OPTIONS if event_type in selected]
                # 自动同步检测开关
                self._sync_detection_flags(notify_cfg["enabled_alert_types"])
                # 保存配置
                if self.save_config():
                    print(f"Enabled Feishu alert types: {notify_cfg['enabled_alert_types']}")
                cv2.destroyWindow("Notification Alerts")
                return True
            if key == ord('q'):
                print("Skipped notification alert selection")
                cv2.destroyWindow("Notification Alerts")
                return False
            if key == ord('a'):
                selected.update(default_enabled)
            elif key == ord('n'):
                selected.clear()
            elif ord('1') <= key <= ord(str(len(ALERT_OPTIONS))):
                event_type = ALERT_OPTIONS[key - ord('1')][0]
                if event_type in selected:
                    selected.remove(event_type)
                else:
                    selected.add(event_type)

    def run(self):
        """运行校准工具"""
        print("\nBaby Region Calibrator")
        print("=" * 50)
        print("Controls:")
        print("1. Left click four corners around the monitored area")
        print("2. The region is auto-completed after the 4th point")
        print("3. Clicking a 5th point clears the old region and starts a new one")
        print("4. Right click/u: undo, r: reset, s: save, q: quit")
        print("=" * 50)

        if not self.init_camera():
            return 1

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.frame = self.read_frame()
            self.height, self.width = self.frame.shape[:2]
            print(f"Calibration frame size: {self.width}x{self.height}")
            self._draw_polygon()
            cv2.imshow(self.window_name, self.temp_frame)
            cv2.waitKey(100)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
        except Exception as e:
            print(f"Failed to create preview window: {e}")
            print("Please run this tool from the Raspberry Pi desktop session, not a headless SSH shell.")
            self.close_camera()
            return 1

        try:
            while True:
                self.frame = self.read_frame()
                self._draw_polygon()

                if not self.is_completed:
                    help_text = f"Click 4 corners: {len(self.points)}/4"
                    color = (0, 165, 255)
                else:
                    help_text = "Region ready. Press s to save, or click again to redraw"
                    color = (0, 255, 0)

                cv2.putText(self.temp_frame, help_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                help_lines = [
                    "Left click: add corner; 4th point auto-completes",
                    "5th click: redraw | Right click/u: undo | r: reset | s: save | q: quit"
                ]
                y_offset = self.height - 10
                for line in reversed(help_lines):
                    cv2.putText(self.temp_frame, line, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    y_offset -= 22

                cv2.imshow(self.window_name, self.temp_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Canceled, no changes saved")
                    break
                elif key == ord('s'):
                    if self.save_region():
                        print("Calibration saved")
                        self.configure_notification_alerts()
                        break
                elif key == ord('r'):
                    self.points = []
                    self.is_completed = False
                    self.current_region = None
                    print("Reset all points")
                elif key == ord('u') or key == 8 or key == 127:
                    if self.points and not self.is_completed:
                        removed = self.points.pop()
                        print(f"Undo point: {removed}, remaining {len(self.points)}")

        finally:
            self.close_camera()
            cv2.destroyAllWindows()

        return 0


def main():
    calibrator = RegionCalibrator()
    return calibrator.run()


if __name__ == "__main__":
    sys.exit(main())
