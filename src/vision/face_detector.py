"""
人脸和表情检测模块
使用 MediaPipe Face Detection 和自定义表情分类实现哭闹检测
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Dict, Optional


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 适合近距离，1 适合远距离
            min_detection_confidence=min_detection_confidence
        )

        # 表情检测的关键点索引
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        self.MOUTH_TOP = 0
        self.MOUTH_BOTTOM = 17
        self.EYE_LEFT_TOP = 159
        self.EYE_LEFT_BOTTOM = 145
        self.EYE_RIGHT_TOP = 386
        self.EYE_RIGHT_BOTTOM = 374
        self.EYEBROW_LEFT = 70
        self.EYEBROW_RIGHT = 300

        # 加载人脸关键点检测
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """检测人脸"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)

                faces.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": detection.score[0],
                    "keypoints": detection.location_data.relative_keypoints
                })

        return faces

    def detect_face_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """检测人脸关键点"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            landmark_points = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark])
            return landmark_points

        return None

    def detect_cry_expression(self, landmarks: np.ndarray) -> Tuple[float, Dict]:
        """检测哭闹表情
        返回: (confidence, features)
        confidence: 0-1，哭闹的置信度
        features: 检测到的表情特征
        """
        if landmarks is None or len(landmarks) < 478:
            return 0.0, {}

        features = {}

        # 1. 嘴巴张开程度
        mouth_left = landmarks[self.MOUTH_LEFT]
        mouth_right = landmarks[self.MOUTH_RIGHT]
        mouth_top = landmarks[self.MOUTH_TOP]
        mouth_bottom = landmarks[self.MOUTH_BOTTOM]

        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
        mouth_aspect_ratio = mouth_height / max(mouth_width, 1)
        features["mouth_aspect_ratio"] = mouth_aspect_ratio

        # 2. 眼睛闭合程度
        eye_left_top = landmarks[self.EYE_LEFT_TOP]
        eye_left_bottom = landmarks[self.EYE_LEFT_BOTTOM]
        eye_right_top = landmarks[self.EYE_RIGHT_TOP]
        eye_right_bottom = landmarks[self.EYE_RIGHT_BOTTOM]

        left_eye_height = np.linalg.norm(eye_left_bottom - eye_left_top)
        right_eye_height = np.linalg.norm(eye_right_bottom - eye_right_top)
        eye_aspect_ratio = (left_eye_height + right_eye_height) / 2.0
        features["eye_aspect_ratio"] = eye_aspect_ratio

        # 3. 眉毛位置
        eyebrow_left = landmarks[self.EYEBROW_LEFT]
        eyebrow_right = landmarks[self.EYEBROW_RIGHT]
        eye_left = landmarks[self.EYE_LEFT_TOP]
        eye_right = landmarks[self.EYE_RIGHT_TOP]

        left_eyebrow_distance = np.linalg.norm(eyebrow_left - eye_left)
        right_eyebrow_distance = np.linalg.norm(eyebrow_right - eye_right)
        eyebrow_height = (left_eyebrow_distance + right_eyebrow_distance) / 2.0
        features["eyebrow_height"] = eyebrow_height

        # 4. 哭闹特征评分
        # 哭闹特征: 嘴巴张大、眼睛紧闭、眉毛皱起
        cry_score = 0.0

        # 嘴巴张开得分 (张嘴越大分数越高)
        if mouth_aspect_ratio > 0.3:
            cry_score += min(0.4, (mouth_aspect_ratio - 0.3) * 2)
        elif mouth_aspect_ratio < 0.1:  # 噘嘴也是哭闹特征
            cry_score += 0.2

        # 眼睛闭合得分 (眼睛越小分数越高)
        if eye_aspect_ratio < 5:
            cry_score += min(0.4, (5 - eye_aspect_ratio) * 0.1)

        # 眉毛皱起得分 (眉毛越高分数越高)
        if eyebrow_height > 10:
            cry_score += min(0.2, (eyebrow_height - 10) * 0.02)

        # 综合置信度
        confidence = min(1.0, cry_score)

        # 特征标签
        features["is_mouth_open"] = mouth_aspect_ratio > 0.3
        features["is_eyes_closed"] = eye_aspect_ratio < 5
        features["is_eyebrow_raised"] = eyebrow_height > 10

        return confidence, features

    def detect_occlusion(self, frame: np.ndarray, landmarks: np.ndarray) -> Tuple[float, Dict]:
        """检测口鼻是否被遮挡
        返回: (confidence, features)
        confidence: 0-1，遮挡的置信度
        """
        if landmarks is None:
            return 0.0, {}

        h, w, _ = frame.shape

        # 获取口鼻区域
        nose_tip = landmarks[1]  # 鼻尖
        mouth_left = landmarks[self.MOUTH_LEFT]
        mouth_right = landmarks[self.MOUTH_RIGHT]
        mouth_bottom = landmarks[self.MOUTH_BOTTOM]

        # 计算口鼻区域边界
        x1 = int(max(0, min(mouth_left[0], nose_tip[0]) - 20))
        y1 = int(max(0, nose_tip[1] - 20))
        x2 = int(min(w, max(mouth_right[0], nose_tip[0]) + 20))
        y2 = int(min(h, mouth_bottom[1] + 20))

        if x1 >= x2 or y1 >= y2:
            return 0.0, {}

        # 提取口鼻区域
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0, {}

        features = {}

        # 1. 肤色检测 - 遮挡区域肤色占比低
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 肤色范围
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / (skin_mask.size + 1e-6)
        features["skin_ratio"] = skin_ratio

        # 2. 纹理检测 - 遮挡区域通常纹理更均匀
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
        texture_variance = np.var(laplacian)
        features["texture_variance"] = texture_variance

        # 3. 边缘检测 - 遮挡区域边缘更少
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.size + 1e-6)
        features["edge_density"] = edge_density

        # 综合遮挡评分
        occlusion_score = 0.0

        # 肤色占比越低，遮挡可能性越高
        if skin_ratio < 0.3:
            occlusion_score += min(0.5, (0.3 - skin_ratio) * 2)

        # 纹理方差越低，遮挡可能性越高（被褥等材质纹理均匀）
        if texture_variance < 100:
            occlusion_score += min(0.3, (100 - texture_variance) / 100)

        # 边缘密度越低，遮挡可能性越高
        if edge_density < 0.05:
            occlusion_score += min(0.2, (0.05 - edge_density) * 10)

        confidence = min(1.0, occlusion_score)
        features["roi_bbox"] = (x1, y1, x2, y2)

        return confidence, features

    def close(self):
        """释放资源"""
        self.face_detection.close()
        self.face_mesh.close()
