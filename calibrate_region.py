#!/usr/bin/env python3
"""
安全区域校准工具
允许用户通过鼠标框选设置婴儿床的安全区域
"""
import sys
import os
import cv2
import numpy as np
import yaml

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import BASE_DIR, CONFIG_PATH
from picamera2 import Picamera2


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

        # 初始化摄像头
        self.picam2 = None
        self.width = self.config["camera"].get("width", 640)
        self.height = self.config["camera"].get("height", 480)
        self.fps = self.config["camera"].get("fps", 15)

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
        """初始化摄像头"""
        print("正在初始化摄像头...")
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameRate": self.fps}
            )
            self.picam2.configure(config)
            self.picam2.start()
            # 预热
            import time
            time.sleep(2)
            print("摄像头初始化完成")
            return True
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            return False

    def save_region(self):
        """保存区域到配置文件"""
        if not self.current_region or len(self.current_region) < 4:
            print("Please select 4 corners before saving")
            return False

        # 更新配置（保存为多边形顶点数组）
        self.config["detection"]["safe_region"] = [[int(p[0]), int(p[1])] for p in self.current_region]

        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"Safe region saved to: {CONFIG_PATH}")
            print(f"New region: {len(self.current_region)} corners")
            return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False

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
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.frame = self.picam2.capture_array()
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
            self._draw_polygon()
            cv2.imshow(self.window_name, self.temp_frame)
            cv2.waitKey(100)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
        except Exception as e:
            print(f"Failed to create preview window: {e}")
            print("Please run this tool from the Raspberry Pi desktop session, not a headless SSH shell.")
            self.picam2.stop()
            return 1

        try:
            while True:
                self.frame = self.picam2.capture_array()
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
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
            self.picam2.stop()
            cv2.destroyAllWindows()

        return 0


def main():
    calibrator = RegionCalibrator()
    return calibrator.run()


if __name__ == "__main__":
    sys.exit(main())
