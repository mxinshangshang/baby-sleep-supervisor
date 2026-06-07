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
        self.min_detection_confidence = min_detection_confidence
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 适合近距离
            min_detection_confidence=min_detection_confidence
        )
        self.face_detection_far = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 对更远/更小的人脸更稳，作为补充
            min_detection_confidence=max(0.25, min_detection_confidence * 0.7)
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
        # ROI/侧脸兜底：全图跟踪失败时，对头部裁剪图用 static_image_mode 低阈值重试。
        self.face_mesh_roi = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=max(0.20, min_detection_confidence * 0.6),
            min_tracking_confidence=0.5
        )

        # 成人手遮挡是婴儿口鼻风险里的高优先级场景。FaceMesh 被手挡住时
        # 往往直接失败，所以需要单独检测手与头脸区域是否重叠。
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=max(0.25, min_detection_confidence * 0.7),
            min_tracking_confidence=0.5,
        )

    def _clip_bbox(self, bbox, w: int, h: int):
        x1, y1, x2, y2 = bbox
        return (max(0, int(x1)), max(0, int(y1)), min(w - 1, int(x2)), min(h - 1, int(y2)))

    def _bbox_iou(self, a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    def _detections_to_faces(self, detections, frame_shape, source: str) -> List[Dict]:
        faces = []
        if not detections:
            return faces
        h, w, _ = frame_shape
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)
            x1, y1, x2, y2 = self._clip_bbox((x1, y1, x2, y2), w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(detection.score[0]),
                "keypoints": detection.location_data.relative_keypoints,
                "source": source,
            })
        return faces

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """检测人脸。用近距+远距两个 BlazeFace 模型互补，并去重。"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = []
        for detector, source in ((self.face_detection, "blazeface_near"), (self.face_detection_far, "blazeface_far")):
            results = detector.process(rgb_frame)
            faces.extend(self._detections_to_faces(results.detections, frame.shape, source))

        # 去重：保留置信度更高/面积更大的框
        faces.sort(key=lambda f: (f["confidence"], (f["bbox"][2]-f["bbox"][0])*(f["bbox"][3]-f["bbox"][1])), reverse=True)
        deduped = []
        for face in faces:
            if all(self._bbox_iou(face["bbox"], kept["bbox"]) < 0.45 for kept in deduped):
                deduped.append(face)
        return deduped

    def _landmarks_from_results(self, results, shape) -> Optional[np.ndarray]:
        if not results.multi_face_landmarks:
            return None
        h, w = shape[:2]
        landmarks = results.multi_face_landmarks[0]
        return np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark], dtype=np.float32)

    def _expand_bbox(self, bbox, frame_shape, scale: float = 1.9):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        size = max(bw, bh) * scale
        nx1, ny1 = int(cx - size / 2), int(cy - size / 2)
        nx2, ny2 = int(cx + size / 2), int(cy + size / 2)
        return self._clip_bbox((nx1, ny1, nx2, ny2), w, h)

    def _rotate_image(self, image: np.ndarray, angle: float):
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        inv = cv2.invertAffineTransform(m)
        return rotated, inv

    def _try_mesh(self, bgr_image: np.ndarray, roi_mode: bool = False) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        detector = self.face_mesh_roi if roi_mode else self.face_mesh
        return self._landmarks_from_results(detector.process(rgb), bgr_image.shape)

    def detect_face_landmarks(self, frame: np.ndarray, head_bbox=None) -> Optional[np.ndarray]:
        """检测人脸关键点。

        先跑全图 FaceMesh；失败后，如果姿态模型提供了 head_bbox，就对头部 ROI 进行
        扩框、放大、多角度旋转后重试。这样能覆盖婴儿床俯视侧脸/微侧脸的常见情况。
        返回坐标始终映射回原图。
        """
        # 1) 全图常规检测，适合正脸/微侧脸。
        full_landmarks = self._try_mesh(frame, roi_mode=False)
        if full_landmarks is not None:
            return full_landmarks

        if head_bbox is None:
            return None

        # 2) 头部 ROI 兜底：扩大头框，旋转重试。角度不宜太多，避免树莓派过载。
        x1, y1, x2, y2 = self._expand_bbox(head_bbox, frame.shape, scale=2.1)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 40 or roi.shape[1] < 40:
            return None

        # 放大 ROI，给 FaceMesh 更大的脸部像素。
        target = 256
        scale = max(1.0, target / float(max(roi.shape[:2])))
        roi_big = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else roi

        for angle in (0, -20, 20, -35, 35, -50, 50):
            rotated, inv = self._rotate_image(roi_big, angle)
            lm = self._try_mesh(rotated, roi_mode=True)
            if lm is None:
                continue
            # 映射：rotated -> roi_big -> 原图
            ones = np.ones((lm.shape[0], 1), dtype=np.float32)
            pts = np.hstack([lm.astype(np.float32), ones]) @ inv.T
            pts = pts / scale
            pts[:, 0] += x1
            pts[:, 1] += y1
            return pts.astype(np.float32)

        return None

    def classify_face_orientation(self, landmarks: Optional[np.ndarray]) -> Dict:
        """基于 FaceMesh 关键点粗分正脸/微侧脸/侧脸。

        yaw_ratio 约等于鼻尖相对双眼中心的水平偏移，数值越大越侧。
        这是近似分类，不是精确 3D 姿态估计。成功的前提是 FaceMesh 已经给出关键点。
        """
        if landmarks is None or len(landmarks) < 292:
            return {"available": False, "orientation": "unknown", "yaw_ratio": None, "direction": "unknown"}
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose = landmarks[1]
        mouth_l = landmarks[61]
        mouth_r = landmarks[291]
        eye_center = (left_eye + right_eye) / 2.0
        face_width = max(1.0, float(np.linalg.norm(right_eye - left_eye)))
        yaw_ratio = float((nose[0] - eye_center[0]) / face_width)
        mouth_center = (mouth_l + mouth_r) / 2.0
        # mouth/nose disagreement hints extreme side/poor mesh; include but keep simple.
        direction = "right" if yaw_ratio > 0.18 else "left" if yaw_ratio < -0.18 else "center"
        abs_yaw = abs(yaw_ratio)
        if abs_yaw < 0.18:
            orientation = "front"
        elif abs_yaw < 0.38:
            orientation = "slight_side"
        else:
            orientation = "side"
        return {
            "available": True,
            "orientation": orientation,
            "yaw_ratio": yaw_ratio,
            "direction": direction,
            "mouth_nose_offset": float((nose[0] - mouth_center[0]) / max(1.0, face_width)),
        }

    def detect_hands(self, frame: np.ndarray) -> List[Dict]:
        """Detect hand bounding boxes. Used as a high-risk occluder near baby face."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        hands = []
        if not results.multi_hand_landmarks:
            return hands
        h, w = frame.shape[:2]
        for hand_landmarks in results.multi_hand_landmarks:
            pts = np.array([(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark], dtype=np.float32)
            x1, y1 = np.min(pts, axis=0)
            x2, y2 = np.max(pts, axis=0)
            pad = 12
            bbox = self._clip_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), w, h)
            hands.append({
                "bbox": bbox,
                "center": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                "landmarks": pts,
                "confidence": 0.70,
            })
        return hands

    def detect_hands_near_head(self, frame: np.ndarray, head_bbox) -> List[Dict]:
        """Detect hands only around the head/face ROI to reduce CPU and false positives."""
        if head_bbox is None:
            return []
        x1, y1, x2, y2 = self._expand_bbox_pixels(head_bbox, frame.shape, pad_ratio=1.35, min_pad=70)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 40 or roi.shape[1] < 40:
            return []
        hands = self.detect_hands(roi)
        mapped = []
        for hand in hands:
            hb = hand.get("bbox")
            if not hb:
                continue
            mx1, my1, mx2, my2 = hb[0] + x1, hb[1] + y1, hb[2] + x1, hb[3] + y1
            hand["bbox"] = self._clip_bbox((mx1, my1, mx2, my2), frame.shape[1], frame.shape[0])
            if "center" in hand:
                hand["center"] = (hand["center"][0] + x1, hand["center"][1] + y1)
            if "landmarks" in hand:
                hand["landmarks"] = hand["landmarks"] + np.array([x1, y1], dtype=np.float32)
            hand["source"] = "head_roi"
            mapped.append(hand)
        return mapped

    def _expand_bbox_pixels(self, bbox, frame_shape, pad_ratio: float = 0.35, min_pad: int = 18):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        pad = max(min_pad, int(max(x2 - x1, y2 - y1) * pad_ratio))
        return self._clip_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), w, h)

    def score_hands_against_roi(self, roi_bbox, hands: List[Dict]) -> Tuple[float, Dict]:
        """Score hand/occluder overlap against an exact mouth-nose ROI."""
        if not roi_bbox:
            return 0.0, {"reason": "no_roi"}
        rx1, ry1, rx2, ry2 = roi_bbox
        roi_area = max(1, (rx2 - rx1) * (ry2 - ry1))
        best_overlap = 0.0
        best_center_distance = 999.0
        risky = []
        roi_center = np.array([(rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0], dtype=np.float32)
        roi_scale = max(1.0, float(np.hypot(rx2 - rx1, ry2 - ry1)))
        for hand in hands or []:
            hb = hand.get("bbox")
            if not hb:
                continue
            ix1, iy1 = max(rx1, hb[0]), max(ry1, hb[1])
            ix2, iy2 = min(rx2, hb[2]), min(ry2, hb[3])
            overlap = 0.0 if ix2 <= ix1 or iy2 <= iy1 else ((ix2 - ix1) * (iy2 - iy1)) / float(roi_area)
            hc = np.array([(hb[0] + hb[2]) / 2.0, (hb[1] + hb[3]) / 2.0], dtype=np.float32)
            dist = float(np.linalg.norm(hc - roi_center) / roi_scale)
            best_overlap = max(best_overlap, overlap)
            best_center_distance = min(best_center_distance, dist)
            # Mesh is available here, so be strict: nearby hands/arms are common and should
            # not be treated as airway occlusion unless they clearly cover the exact mouth/nose ROI.
            if overlap >= 0.22 or (overlap >= 0.12 and dist < 0.35):
                risky.append({"bbox": hb, "mouth_nose_overlap": float(overlap), "center_distance": float(dist)})
        if risky:
            return min(1.0, 0.68 + best_overlap * 1.6), {
                "reason": "hand_overlaps_mouth_nose_roi",
                "hand_mouth_nose_overlap": float(best_overlap),
                "hand_mouth_nose_center_distance": float(best_center_distance),
                "overlapping_hands": risky,
            }
        return 0.0, {
            "reason": "no_hand_overlap_mouth_nose_roi",
            "hand_mouth_nose_overlap": float(best_overlap),
            "hand_mouth_nose_center_distance": float(best_center_distance),
            "overlapping_hands": [],
        }

    def detect_head_occlusion_fallback(self, frame: np.ndarray, head_bbox, hands: List[Dict]) -> Tuple[float, Dict]:
        """Fallback occlusion detector when FaceMesh/mouth-nose landmarks are unavailable.

        Conservative rule: a hand near the broader head box is not enough. It must overlap
        a smaller airway ROI around the lower/central face, or its center must be very close
        to that airway ROI. This avoids false positives from hands/arms near the body.
        """
        if head_bbox is None:
            return 0.0, {"reason": "no_head_bbox"}

        x1, y1, x2, y2 = head_bbox
        hw, hh = max(1, x2 - x1), max(1, y2 - y1)
        # Conservative airway ROI: central/lower part of the head box, slightly expanded.
        ax1 = int(x1 + 0.18 * hw)
        ax2 = int(x2 - 0.18 * hw)
        ay1 = int(y1 + 0.30 * hh)
        ay2 = int(y2 + 0.15 * hh)
        ax1, ay1, ax2, ay2 = self._clip_bbox((ax1 - 8, ay1 - 8, ax2 + 8, ay2 + 8), frame.shape[1], frame.shape[0])
        airway_area = max(1, (ax2 - ax1) * (ay2 - ay1))

        best_airway_overlap = 0.0
        best_center_distance = 999.0
        risky_hands = []
        airway_center = np.array([(ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0], dtype=np.float32)
        airway_scale = max(1.0, float(np.hypot(ax2 - ax1, ay2 - ay1)))

        for hand in hands or []:
            hb = hand.get("bbox")
            if not hb:
                continue
            ix1, iy1 = max(ax1, hb[0]), max(ay1, hb[1])
            ix2, iy2 = min(ax2, hb[2]), min(ay2, hb[3])
            airway_overlap = 0.0 if ix2 <= ix1 or iy2 <= iy1 else ((ix2 - ix1) * (iy2 - iy1)) / float(airway_area)
            hc = np.array([(hb[0] + hb[2]) / 2.0, (hb[1] + hb[3]) / 2.0], dtype=np.float32)
            center_dist = float(np.linalg.norm(hc - airway_center) / airway_scale)
            best_airway_overlap = max(best_airway_overlap, airway_overlap)
            best_center_distance = min(best_center_distance, center_dist)
            if airway_overlap >= 0.08 or (airway_overlap >= 0.03 and center_dist < 0.55):
                risky_hands.append({
                    "bbox": hb,
                    "airway_overlap": float(airway_overlap),
                    "center_distance": float(center_dist),
                })

        roi = frame[ay1:ay2, ax1:ax2]
        features = {
            "roi_bbox": (ax1, ay1, ax2, ay2),
            "airway_roi_bbox": (ax1, ay1, ax2, ay2),
            "head_bbox": head_bbox,
            "hand_airway_overlap": float(best_airway_overlap),
            "hand_airway_center_distance": float(best_center_distance),
            "overlapping_hands": risky_hands,
            "mode": "airway_roi_hand_fallback",
        }

        if roi.size:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            features["texture_variance"] = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))
            edges = cv2.Canny(gray, 50, 150)
            features["edge_density"] = float(np.sum(edges > 0) / (edges.size + 1e-6))

        if risky_hands:
            confidence = min(1.0, 0.70 + best_airway_overlap * 2.2)
            features["reason"] = "hand_overlaps_airway_roi"
            return confidence, features

        features["reason"] = "face_mesh_unavailable_head_visible_airway_uncertain"
        return 0.38, features

    def detect_mouth_open_fallback(self, frame: np.ndarray, head_bbox) -> Tuple[float, Dict]:
        """Fallback mouth-open cue when FaceMesh is unavailable.

        It looks for a dark/open-mouth-like blob in the lower-middle part of the head ROI.
        This is intentionally conservative and only supplies a weak/medium cue for distress fusion.
        """
        if head_bbox is None:
            return 0.0, {"reason": "no_head_bbox"}
        x1, y1, x2, y2 = head_bbox
        hw, hh = max(1, x2 - x1), max(1, y2 - y1)
        # Lower/central face approximation. Works as a fallback only; FaceMesh is preferred.
        mx1 = int(x1 + 0.20 * hw)
        mx2 = int(x2 - 0.20 * hw)
        my1 = int(y1 + 0.38 * hh)
        my2 = int(y2 + 0.12 * hh)
        mx1, my1, mx2, my2 = self._clip_bbox((mx1 - 6, my1 - 6, mx2 + 6, my2 + 6), frame.shape[1], frame.shape[0])
        roi = frame[my1:my2, mx1:mx2]
        if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
            return 0.0, {"reason": "empty_mouth_roi", "roi_bbox": (mx1, my1, mx2, my2)}

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Open mouth often appears as a darker, moderately saturated region.
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        dark = v < max(45, np.percentile(v, 28))
        saturated = s > 25
        mask = (dark & saturated).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = 0.0
        best_bbox = None
        roi_area = float(mask.size)
        for c in contours:
            area = cv2.contourArea(c)
            if area < max(4, roi_area * 0.015):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect = bh / max(1, bw)
            area_ratio = area / roi_area
            center_bias = 1.0 - min(1.0, abs((bx + bw / 2) - roi.shape[1] / 2) / max(1, roi.shape[1] / 2))
            shape_score = max(0.0, min(1.0, area_ratio * 12.0)) * 0.55 + max(0.0, min(1.0, aspect / 1.2)) * 0.25 + center_bias * 0.20
            if shape_score > best_score:
                best_score = shape_score
                best_bbox = (mx1 + bx, my1 + by, mx1 + bx + bw, my1 + by + bh)

        dark_ratio = float(np.sum(mask > 0) / (mask.size + 1e-6))
        contrast = float(np.std(gray) / 64.0)
        score = min(0.75, max(best_score, min(0.55, dark_ratio * 4.0 + contrast * 0.12)))
        return float(score), {
            "reason": "mouth_open_fallback_no_mesh",
            "roi_bbox": (mx1, my1, mx2, my2),
            "candidate_bbox": best_bbox,
            "dark_ratio": dark_ratio,
            "contrast": contrast,
            "mouth_open_score": float(score),
        }

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
        features["mouth_aspect_ratio"] = float(mouth_aspect_ratio)
        features["mouth_width"] = float(mouth_width)
        features["mouth_height"] = float(mouth_height)
        # Side-face cry detection: mouth opening is the most useful visual cue, but
        # needs temporal/motion support in fusion to avoid single-frame yawning/noise.
        mouth_open_score = max(0.0, min(1.0, (mouth_aspect_ratio - 0.22) / 0.38))
        features["mouth_open_score"] = float(mouth_open_score)

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

        # 嘴巴张开得分 (张嘴越大分数越高)。婴儿哭闹时嘴部大张是强视觉线索。
        if mouth_aspect_ratio > 0.28:
            cry_score += min(0.55, (mouth_aspect_ratio - 0.28) * 2.4)
        elif mouth_aspect_ratio < 0.1:  # 噘嘴也是哭闹特征
            cry_score += 0.15

        # 眼睛闭合得分 (眼睛越小分数越高)
        if eye_aspect_ratio < 5:
            cry_score += min(0.4, (5 - eye_aspect_ratio) * 0.1)

        # 眉毛皱起得分 (眉毛越高分数越高)
        if eyebrow_height > 10:
            cry_score += min(0.2, (eyebrow_height - 10) * 0.02)

        # 综合置信度
        confidence = min(1.0, cry_score)

        # 特征标签
        features["is_mouth_open"] = bool(mouth_aspect_ratio > 0.28)
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

        # 综合遮挡评分。注意：FaceMesh 已经可用时，说明模型仍看到了口鼻附近结构；
        # 单纯肤色低/纹理低在绿光、花被子、侧脸下很容易误报，所以先给保守分。
        skin_evidence = max(0.0, min(1.0, (0.28 - skin_ratio) / 0.28))
        texture_evidence = max(0.0, min(1.0, (80 - texture_variance) / 80))
        edge_evidence = max(0.0, min(1.0, (0.035 - edge_density) / 0.035))

        visual_score = 0.45 * skin_evidence + 0.35 * texture_evidence + 0.20 * edge_evidence
        # No explicit occluder: cap below danger threshold. Fallback/hand-overlap logic may raise it.
        confidence = min(0.48, visual_score * 0.62)

        features["roi_bbox"] = (x1, y1, x2, y2)
        features["skin_evidence"] = float(skin_evidence)
        features["texture_evidence"] = float(texture_evidence)
        features["edge_evidence"] = float(edge_evidence)
        features["visual_score_raw"] = float(visual_score)
        features["reason"] = "mesh_visible_visual_only_capped"

        return confidence, features

    def close(self):
        """释放资源"""
        self.face_detection.close()
        self.face_detection_far.close()
        self.face_mesh.close()
        self.face_mesh_roi.close()
        self.hands.close()
