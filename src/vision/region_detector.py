"""
区域检测模块
用于检测婴儿是否在指定的安全区域内
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional


class RegionDetector:
    def __init__(self):
        self.safe_region = None  # ((x1, y1), (x2, y2))
        self.region_mask = None
        self.motion_threshold = 25
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    def set_safe_region(self, region: Tuple[Tuple[int, int], Tuple[int, int]]):
        """设置安全区域"""
        self.safe_region = region
        (x1, y1), (x2, y2) = region
        self.region_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.rectangle(self.region_mask, (0, 0), (x2 - x1, y2 - y1), 255, -1)

    def point_in_region(self, point: Tuple[int, int]) -> bool:
        """检查点是否在安全区域内"""
        if self.safe_region is None:
            return True  # 未设置区域默认都在里面

        (x1, y1), (x2, y2) = self.safe_region
        return x1 <= point[0] <= x2 and y1 <= point[1] <= y2

    def bbox_overlap_with_region(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, bool]:
        """计算边界框与安全区域的重叠度
        返回: (overlap_ratio, is_mostly_inside)
        """
        if self.safe_region is None:
            return 1.0, True

        (rx1, ry1), (rx2, ry2) = self.safe_region
        bx1, by1, bx2, by2 = bbox

        # 计算交集
        x1 = max(rx1, bx1)
        y1 = max(ry1, by1)
        x2 = min(rx2, bx2)
        y2 = min(ry2, by2)

        if x1 >= x2 or y1 >= y2:
            return 0.0, False

        intersection_area = (x2 - x1) * (y2 - y1)
        bbox_area = (bx2 - bx1) * (by2 - by1)

        if bbox_area == 0:
            return 0.0, False

        overlap_ratio = intersection_area / bbox_area
        is_mostly_inside = overlap_ratio > 0.7

        return overlap_ratio, is_mostly_inside

    def detect_motion_in_region(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """检测安全区域内的运动
        返回: (has_motion, features)
        """
        if self.safe_region is None:
            return False, {}

        (x1, y1), (x2, y2) = self.safe_region
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return False, {}

        # 背景减除
        fg_mask = self.background_subtractor.apply(roi)

        # 去除阴影
        fg_mask[fg_mask == 127] = 0

        # 形态学操作，去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # 计算运动像素比例
        motion_pixels = np.sum(fg_mask > 0)
        total_pixels = fg_mask.size
        motion_ratio = motion_pixels / (total_pixels + 1e-6)

        has_motion = motion_ratio > 0.01  # 超过1%像素变化认为有运动

        features = {
            "motion_ratio": motion_ratio,
            "motion_pixels": int(motion_pixels),
            "has_significant_motion": has_motion
        }

        return has_motion, features

    def draw_region(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """在帧上绘制安全区域"""
        if self.safe_region is None:
            return frame

        (x1, y1), (x2, y2) = self.safe_region
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, "Safe Region", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame
