"""
人体和肢体检测模块
使用 MediaPipe Pose 或 YOLO 实现人体关键点检测，用于踢被子和肢体裸露检测
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Dict, Optional


class BodyDetector:
    def __init__(self, min_detection_confidence: float = 0.5, model_complexity: int = 1):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

        # 关键点索引
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24
        self.LEFT_KNEE = 25
        self.RIGHT_KNEE = 26
        self.LEFT_ANKLE = 27
        self.RIGHT_ANKLE = 28

        # 肤色检测阈值
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    def detect_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """检测人体姿态
        返回包含关键点、分割掩码等信息的字典
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return None

        h, w, _ = frame.shape

        # 转换关键点坐标
        landmarks = np.array([
            (lm.x * w, lm.y * h, lm.visibility)
            for lm in results.pose_landmarks.landmark
        ])

        # 生成分割掩码
        segmentation_mask = None
        if results.segmentation_mask is not None:
            segmentation_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            segmentation_mask = cv2.resize(segmentation_mask, (w, h))

        return {
            "landmarks": landmarks,
            "segmentation_mask": segmentation_mask,
            "pose_world_landmarks": results.pose_world_landmarks
        }

    def detect_limb_exposure(self, frame: np.ndarray, pose_data: Dict) -> Tuple[float, Dict]:
        """检测肢体裸露程度（踢被子）
        返回: (exposure_ratio, features)
        exposure_ratio: 0-1，裸露皮肤占身体区域的比例
        """
        if pose_data is None or pose_data["segmentation_mask"] is None:
            return 0.0, {}

        h, w, _ = frame.shape
        segmentation_mask = pose_data["segmentation_mask"]
        landmarks = pose_data["landmarks"]

        # 获取身体区域
        body_pixels = segmentation_mask > 0
        if np.sum(body_pixels) == 0:
            return 0.0, {}

        # 在身体区域内检测肤色
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv_frame, self.lower_skin, self.upper_skin)

        # 只考虑身体区域内的肤色
        body_skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=segmentation_mask)

        # 计算裸露比例
        skin_pixels = np.sum(body_skin_mask > 0)
        total_body_pixels = np.sum(body_pixels)
        exposure_ratio = skin_pixels / (total_body_pixels + 1e-6)

        features = {
            "exposure_ratio": exposure_ratio,
            "skin_pixels": int(skin_pixels),
            "body_pixels": int(total_body_pixels)
        }

        # 检测哪些肢体裸露
        exposed_limbs = []
        limb_threshold = 0.5  # 肢体区域肤色占比超过阈值认为裸露

        # 手臂检测
        left_arm_visible = landmarks[self.LEFT_WRIST][2] > 0.5 and landmarks[self.LEFT_ELBOW][2] > 0.5
        right_arm_visible = landmarks[self.RIGHT_WRIST][2] > 0.5 and landmarks[self.RIGHT_ELBOW][2] > 0.5

        if left_arm_visible:
            # 检查左手臂区域肤色
            x1, y1 = int(max(0, landmarks[self.LEFT_WRIST][0] - 20)), int(max(0, landmarks[self.LEFT_WRIST][1] - 20))
            x2, y2 = int(min(w, landmarks[self.LEFT_ELBOW][0] + 20)), int(min(h, landmarks[self.LEFT_ELBOW][1] + 20))
            arm_roi = body_skin_mask[y1:y2, x1:x2]
            if arm_roi.size > 0 and np.sum(arm_roi > 0) / arm_roi.size > limb_threshold:
                exposed_limbs.append("left_arm")

        if right_arm_visible:
            x1, y1 = int(max(0, landmarks[self.RIGHT_WRIST][0] - 20)), int(max(0, landmarks[self.RIGHT_WRIST][1] - 20))
            x2, y2 = int(min(w, landmarks[self.RIGHT_ELBOW][0] + 20)), int(min(h, landmarks[self.RIGHT_ELBOW][1] + 20))
            arm_roi = body_skin_mask[y1:y2, x1:x2]
            if arm_roi.size > 0 and np.sum(arm_roi > 0) / arm_roi.size > limb_threshold:
                exposed_limbs.append("right_arm")

        # 腿部检测
        left_leg_visible = landmarks[self.LEFT_ANKLE][2] > 0.5 and landmarks[self.LEFT_KNEE][2] > 0.5
        right_leg_visible = landmarks[self.RIGHT_ANKLE][2] > 0.5 and landmarks[self.RIGHT_KNEE][2] > 0.5

        if left_leg_visible:
            x1, y1 = int(max(0, landmarks[self.LEFT_ANKLE][0] - 20)), int(max(0, landmarks[self.LEFT_ANKLE][1] - 20))
            x2, y2 = int(min(w, landmarks[self.LEFT_KNEE][0] + 20)), int(min(h, landmarks[self.LEFT_KNEE][1] + 20))
            leg_roi = body_skin_mask[y1:y2, x1:x2]
            if leg_roi.size > 0 and np.sum(leg_roi > 0) / leg_roi.size > limb_threshold:
                exposed_limbs.append("left_leg")

        if right_leg_visible:
            x1, y1 = int(max(0, landmarks[self.RIGHT_ANKLE][0] - 20)), int(max(0, landmarks[self.RIGHT_ANKLE][1] - 20))
            x2, y2 = int(min(w, landmarks[self.RIGHT_KNEE][0] + 20)), int(min(h, landmarks[self.RIGHT_KNEE][1] + 20))
            leg_roi = body_skin_mask[y1:y2, x1:x2]
            if leg_roi.size > 0 and np.sum(leg_roi > 0) / leg_roi.size > limb_threshold:
                exposed_limbs.append("right_leg")

        features["exposed_limbs"] = exposed_limbs
        features["limb_count"] = len(exposed_limbs)

        return exposure_ratio, features

    def is_in_region(self, pose_data: Dict, region: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[bool, Dict]:
        """检测人体是否在指定区域内
        region: ((x1, y1), (x2, y2)) 区域的左上角和右下角坐标
        """
        if pose_data is None:
            return False, {}

        landmarks = pose_data["landmarks"]
        (x1, y1), (x2, y2) = region

        # 检查关键点是否在区域内
        visible_landmarks = [lm for lm in landmarks if lm[2] > 0.5]
        if not visible_landmarks:
            return False, {}

        # 计算人体边界框
        min_x = min(lm[0] for lm in visible_landmarks)
        max_x = max(lm[0] for lm in visible_landmarks)
        min_y = min(lm[1] for lm in visible_landmarks)
        max_y = max(lm[1] for lm in visible_landmarks)

        # 计算重叠度
        overlap_x1 = max(min_x, x1)
        overlap_y1 = max(min_y, y1)
        overlap_x2 = min(max_x, x2)
        overlap_y2 = min(max_y, y2)

        if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
            overlap_ratio = 0.0
        else:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            body_area = (max_x - min_x) * (max_y - min_y)
            overlap_ratio = overlap_area / (body_area + 1e-6)

        # 中心点是否在区域内
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_in_region = (x1 <= center_x <= x2) and (y1 <= center_y <= y2)

        is_in = overlap_ratio > 0.7 or center_in_region

        features = {
            "body_bbox": (int(min_x), int(min_y), int(max_x), int(max_y)),
            "center": (int(center_x), int(center_y)),
            "overlap_ratio": overlap_ratio,
            "center_in_region": center_in_region
        }

        return is_in, features

    def close(self):
        """释放资源"""
        self.pose.close()
