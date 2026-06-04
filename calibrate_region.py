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
        self.window_name = "安全区域校准"
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_region = None
        self.frame = None
        self.temp_frame = None

        # 加载现有配置
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 获取现有区域配置
        existing_region = self.config["detection"].get("safe_region", [[50, 50], [590, 430]])
        self.current_region = (tuple(existing_region[0]), tuple(existing_region[1]))
        print(f"当前安全区域: {self.current_region}")

        # 初始化摄像头
        self.picam2 = None
        self.width = self.config["camera"].get("width", 640)
        self.height = self.config["camera"].get("height", 480)
        self.fps = self.config["camera"].get("fps", 15)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            print(f"开始选择区域: {self.start_point}")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.start_point:
                # 绘制临时矩形
                self.temp_frame = self.frame.copy()
                cv2.rectangle(self.temp_frame, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, self.temp_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            print(f"结束选择区域: {self.end_point}")

            # 确保坐标正确（左上，右下）
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])

            # 限制在画面范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.width - 1, x2)
            y2 = min(self.height - 1, y2)

            self.current_region = ((x1, y1), (x2, y2))
            print(f"选择的区域: {self.current_region}")

            # 绘制最终矩形
            self.temp_frame = self.frame.copy()
            cv2.rectangle(self.temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.temp_frame, f"Selected Region: {x1},{y1} - {x2},{y2}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
        if not self.current_region:
            print("没有选择区域")
            return False

        (x1, y1), (x2, y2) = self.current_region

        # 更新配置
        self.config["detection"]["safe_region"] = [[int(x1), int(y1)], [int(x2), int(y2)]]

        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"安全区域已保存到配置文件: {CONFIG_PATH}")
            print(f"新区域: {self.current_region}")
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False

    def run(self):
        """运行校准工具"""
        print("\n安全区域校准工具")
        print("=" * 40)
        print("操作说明:")
        print("1. 用鼠标在画面上拖动框选婴儿床区域")
        print("2. 按 's' 保存选择的区域")
        print("3. 按 'r' 重置选择")
        print("4. 按 'q' 退出不保存")
        print("=" * 40)

        # 初始化摄像头
        if not self.init_camera():
            return 1

        # 创建窗口
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        try:
            while True:
                # 捕获帧
                self.frame = self.picam2.capture_array()
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)

                # 如果没有在绘制，显示当前区域
                if not self.drawing:
                    self.temp_frame = self.frame.copy()
                    if self.current_region:
                        (x1, y1), (x2, y2) = self.current_region
                        cv2.rectangle(self.temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(self.temp_frame, f"Current Region: {x1},{y1} - {x2},{y2}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 显示帮助文字
                    help_lines = [
                        "拖拽鼠标选择区域",
                        "s: 保存  r: 重置  q: 退出"
                    ]
                    y_offset = self.height - 10
                    for line in reversed(help_lines):
                        cv2.putText(self.temp_frame, line, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset -= 25

                # 显示帧
                cv2.imshow(self.window_name, self.temp_frame)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户取消，未保存更改")
                    break
                elif key == ord('s'):
                    if self.save_region():
                        print("区域校准完成，可以退出了")
                        break
                elif key == ord('r'):
                    self.start_point = None
                    self.end_point = None
                    print("已重置选择")

        finally:
            # 释放资源
            self.picam2.stop()
            cv2.destroyAllWindows()

        return 0


def main():
    calibrator = RegionCalibrator()
    return calibrator.run()


if __name__ == "__main__":
    sys.exit(main())
