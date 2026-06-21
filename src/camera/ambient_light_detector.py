"""
环境光线检测器
基于图像帧分析环境亮度，用于双摄像头自动切换决策
支持迟滞比较器防抖，避免频繁切换
"""
import cv2
import numpy as np
import time
from typing import Optional, Tuple


class AmbientLightDetector:
    """从图像帧检测环境光线（无额外硬件）"""

    def __init__(self, config: dict):
        self.night_threshold_low = config.get("night_threshold", 0.15)
        self.night_threshold_high = config.get("day_threshold", 0.35)
        self.stable_frames = config.get("stable_frames", 30)
        self.min_switch_interval = config.get("min_switch_interval", 300)

        self.consecutive_dark = 0
        self.consecutive_bright = 0

        self.brightness_window = []
        self.window_size = 10

    def measure_brightness(self, frame: np.ndarray) -> float:
        """从中心区域测量平均亮度（0~1，归一化到255）"""
        if frame is None or len(frame.shape) < 2:
            return 0.0

        h, w = frame.shape[:2]

        center_x1, center_x2 = w // 4, w * 3 // 4
        center_y1, center_y2 = h // 4, h * 3 // 4

        center_region = frame[center_y1:center_y2, center_x1:center_x2]

        if len(center_region.shape) == 3:
            gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = center_region

        brightness = float(np.mean(gray)) / 255.0

        self.brightness_window.append(brightness)
        if len(self.brightness_window) > self.window_size:
            self.brightness_window.pop(0)

        return sum(self.brightness_window) / len(self.brightness_window)

    def should_switch(self, frame: np.ndarray, current_camera: int,
                      time_since_last_switch: float) -> Optional[int]:
        """判断是否需要切换摄像头。
        返回 None（不切换）、0（切到常规摄像头）或 1（切到夜视摄像头）。
        """
        if time_since_last_switch < self.min_switch_interval:
            return None

        brightness = self.measure_brightness(frame)

        if current_camera == 0:
            # 当前是常规摄像头，检测是否需要切到夜视
            if brightness < self.night_threshold_low:
                self.consecutive_dark += 1
                self.consecutive_bright = 0
                if self.consecutive_dark >= self.stable_frames:
                    return 1
            else:
                self.consecutive_dark = max(0, self.consecutive_dark - 1)
                self.consecutive_bright = 0
        else:
            # 当前是夜视摄像头，检测是否需要切回常规
            if brightness > self.night_threshold_high:
                self.consecutive_bright += 1
                self.consecutive_dark = 0
                if self.consecutive_bright >= self.stable_frames:
                    return 0
            else:
                self.consecutive_bright = max(0, self.consecutive_bright - 1)
                self.consecutive_dark = 0

        return None

    def reset(self):
        """切换后重置计数器，避免历史数据污染新摄像头的检测"""
        self.consecutive_dark = 0
        self.consecutive_bright = 0
        self.brightness_window.clear()
