"""
监督核心模块
整合所有检测逻辑，实现事件判断和告警逻辑
"""
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from src.config import get_config
from src.vision.face_detector import FaceDetector
from src.vision.body_detector import BodyDetector
from src.vision.region_detector import RegionDetector
from src.notifier import Notifier
from src.storage import Storage


class SleepSupervisor:
    def __init__(self):
        self.config = get_config()
        detection_cfg = self.config.get("detection", {})
        supervision_cfg = self.config.get("supervision", {})

        # 初始化检测器
        self.face_detector = FaceDetector(
            min_detection_confidence=self.config["inference"].get("min_detection_confidence", 0.5)
        )
        self.body_detector = BodyDetector(
            min_detection_confidence=self.config["inference"].get("min_detection_confidence", 0.5),
            model_complexity=self.config["inference"].get("model_complexity", 1)
        )
        self.region_detector = RegionDetector()

        # 设置安全区域（支持矩形和多边形两种格式）
        safe_region = detection_cfg.get("safe_region", [[50, 50], [590, 430]])
        if len(safe_region) == 2 and len(safe_region[0]) == 2 and len(safe_region[1]) == 2:
            # 旧版矩形格式
            self.region_detector.set_safe_region((tuple(safe_region[0]), tuple(safe_region[1])))
        else:
            # 新版多边形格式
            self.region_detector.set_safe_region([tuple(p) for p in safe_region])

        # 初始化通知和存储
        self.notifier = Notifier()
        self.storage = Storage()

        # 检测阈值
        self.cry_threshold = detection_cfg.get("cry_confidence_threshold", 0.7)
        self.cry_duration_threshold = detection_cfg.get("cry_duration_threshold", 2.0)
        self.exposure_threshold = detection_cfg.get("exposure_threshold", 0.3)
        self.exposure_duration_threshold = detection_cfg.get("exposure_duration_threshold", 5.0)
        self.occlusion_threshold = detection_cfg.get("occlusion_threshold", 0.6)
        self.occlusion_duration_threshold = detection_cfg.get("occlusion_duration_threshold", 1.0)
        self.region_exit_duration_threshold = detection_cfg.get("region_exit_duration_threshold", 3.0)
        self.pose_visibility_threshold = detection_cfg.get("pose_visibility_threshold", 0.5)
        self.pose_min_visible_landmarks = detection_cfg.get("pose_min_visible_landmarks", 6)
        self.pose_min_core_landmarks = detection_cfg.get("pose_min_core_landmarks", 2)
        self.pose_min_body_bbox_area = detection_cfg.get("pose_min_body_bbox_area", 1800)
        self.pose_min_torso_bbox_area = detection_cfg.get("pose_min_torso_bbox_area", 500)
        self.pose_bbox_padding_px = detection_cfg.get("pose_bbox_padding_px", 12)
        self.head_bbox_padding_px = detection_cfg.get("head_bbox_padding_px", 20)
        self.face_min_bbox_area = detection_cfg.get("face_min_bbox_area", 900)
        self.presence_score_threshold = detection_cfg.get("presence_score_threshold", 0.55)
        self.presence_confirm_ratio = detection_cfg.get("presence_confirm_ratio", 0.6)
        self.presence_uncertain_grace_s = detection_cfg.get("presence_uncertain_grace_s", 2.0)
        self.region_body_overlap_threshold = detection_cfg.get("region_body_overlap_threshold", 0.55)
        self.region_torso_overlap_threshold = detection_cfg.get("region_torso_overlap_threshold", 0.50)
        self.region_exit_confirm_ratio = detection_cfg.get("region_exit_confirm_ratio", 0.7)
        self.alert_requires_confirmed_presence = detection_cfg.get("alert_requires_confirmed_presence", True)
        self.face_absence_enabled = detection_cfg.get("face_absence_detection_enabled", True)
        self.face_absence_duration_threshold = detection_cfg.get("face_absence_duration_threshold", 15.0)

        # 状态跟踪
        self.cry_start_time: Optional[float] = None
        self.exposure_start_time: Optional[float] = None
        self.occlusion_start_time: Optional[float] = None
        self.region_exit_start_time: Optional[float] = None
        self.face_absence_start_time: Optional[float] = None
        self.last_cry_confidence: float = 0.0
        self.last_cry_time: float = 0.0
        self.cry_hold_s: float = detection_cfg.get("cry_hold_s", 8.0)
        self.distress_start_time: Optional[float] = None
        self.distress_threshold = detection_cfg.get("distress_confidence_threshold", 0.55)
        self.distress_duration_threshold = detection_cfg.get("distress_duration_threshold", 1.5)

        # 平滑窗口
        self.cry_confidence_window = deque(maxlen=10)
        self.exposure_ratio_window = deque(maxlen=10)
        self.occlusion_confidence_window = deque(maxlen=10)
        self.presence_score_window = deque(maxlen=detection_cfg.get("presence_window_size", 8))
        self.region_exit_window = deque(maxlen=detection_cfg.get("region_exit_window_size", 8))
        self.head_motion_window = deque(maxlen=detection_cfg.get("motion_window_size", 12))
        self.limb_motion_window = deque(maxlen=detection_cfg.get("motion_window_size", 12))
        self.mouth_open_window = deque(maxlen=detection_cfg.get("mouth_open_window_size", 6))
        self.distress_window = deque(maxlen=detection_cfg.get("distress_window_size", 6))
        self.prev_mouth_open_score: Optional[float] = None
        self.mouth_variation_window = deque(maxlen=detection_cfg.get("mouth_variation_window_size", 6))
        self.prev_motion_points: Optional[Dict] = None
        self.last_confirmed_presence_time = 0.0

        # 检测开关
        self.cry_enabled = detection_cfg.get("cry_detection_enabled", True)
        self.exposure_enabled = detection_cfg.get("limb_exposure_enabled", True)
        self.occlusion_enabled = detection_cfg.get("occlusion_detection_enabled", True)
        self.region_enabled = detection_cfg.get("region_detection_enabled", True)

        # 告警冷却
        self.alert_cooldown = supervision_cfg.get("alert_cooldown_s", 60)
        self.last_alert_time: Dict[str, float] = {}

        # 运行状态
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0

        # 最近的检测结果
        self.last_results: Dict = {}

    def _get_smoothed_value(self, window: deque, new_value: float) -> float:
        """获取平滑后的值"""
        window.append(new_value)
        return sum(window) / len(window) if window else 0.0

    def _should_alert(self, event_type: str) -> bool:
        """检查是否可以发送告警（冷却时间）"""
        now = time.time()
        last_time = self.last_alert_time.get(event_type, 0)
        if now - last_time < self.alert_cooldown:
            return False
        self.last_alert_time[event_type] = now
        return True

    def _bbox_from_points(self, points: List[Tuple[float, float]], frame_shape: Tuple[int, int, int], padding: int) -> Optional[Tuple[int, int, int, int]]:
        if not points:
            return None
        h, w = frame_shape[:2]
        min_x = max(0, int(min(p[0] for p in points) - padding))
        min_y = max(0, int(min(p[1] for p in points) - padding))
        max_x = min(w - 1, int(max(p[0] for p in points) + padding))
        max_y = min(h - 1, int(max(p[1] for p in points) + padding))
        if max_x <= min_x or max_y <= min_y:
            return None
        return min_x, min_y, max_x, max_y

    def _bbox_area(self, bbox: Optional[Tuple[int, int, int, int]]) -> int:
        if not bbox:
            return 0
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    def _bbox_center(self, bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int]]:
        if not bbox:
            return None
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def _summarize_pose(self, pose_data: Optional[Dict], frame_shape: Tuple[int, int, int]) -> Dict:
        summary = {
            "available": False,
            "visible_landmarks": 0,
            "core_landmarks": 0,
            "body_bbox": None,
            "torso_bbox": None,
            "head_bbox": None,
            "body_area": 0,
            "torso_area": 0,
            "body_center": None,
            "torso_center": None,
            "head_center": None,
            "quality": 0.0,
            "segmentation_available": False,
        }
        if not pose_data:
            return summary

        landmarks = pose_data["landmarks"]
        visible = [(lm[0], lm[1]) for lm in landmarks if lm[2] >= self.pose_visibility_threshold]
        head_indices = list(range(0, 11))
        torso_indices = [11, 12, 23, 24]
        core_indices = head_indices + torso_indices
        head_points = [(landmarks[i][0], landmarks[i][1]) for i in head_indices if landmarks[i][2] >= self.pose_visibility_threshold]
        torso_points = [(landmarks[i][0], landmarks[i][1]) for i in torso_indices if landmarks[i][2] >= self.pose_visibility_threshold]
        core_landmarks = sum(1 for i in core_indices if landmarks[i][2] >= self.pose_visibility_threshold)

        body_bbox = self._bbox_from_points(visible, frame_shape, self.pose_bbox_padding_px)
        torso_bbox = self._bbox_from_points(torso_points, frame_shape, self.pose_bbox_padding_px)
        head_bbox = self._bbox_from_points(head_points, frame_shape, self.head_bbox_padding_px)
        body_area = self._bbox_area(body_bbox)
        torso_area = self._bbox_area(torso_bbox)
        segmentation_available = pose_data.get("segmentation_mask") is not None and np.sum(pose_data["segmentation_mask"] > 0) > 500

        score = 0.0
        if len(visible) >= self.pose_min_visible_landmarks and body_area >= self.pose_min_body_bbox_area:
            score += 0.35
        if core_landmarks >= self.pose_min_core_landmarks:
            score += 0.30
        if torso_area >= self.pose_min_torso_bbox_area or head_bbox:
            score += 0.20
        if segmentation_available:
            score += 0.15

        summary.update({
            "available": score > 0,
            "visible_landmarks": len(visible),
            "core_landmarks": core_landmarks,
            "body_bbox": body_bbox,
            "torso_bbox": torso_bbox,
            "head_bbox": head_bbox,
            "body_area": body_area,
            "torso_area": torso_area,
            "body_center": self._bbox_center(body_bbox),
            "torso_center": self._bbox_center(torso_bbox),
            "head_center": self._bbox_center(head_bbox),
            "quality": min(1.0, score),
            "segmentation_available": segmentation_available,
        })
        return summary

    def _summarize_face(self, faces: List[Dict], landmarks) -> Dict:
        summary = {
            "available": False,
            "landmarks_available": landmarks is not None,
            "face_count": len(faces),
            "main_face_bbox": None,
            "main_face_confidence": 0.0,
            "mode": "not_visible",
            "quality": 0.0,
        }
        if not faces:
            if landmarks is not None:
                summary.update({
                    "available": True,
                    "mode": "frontal_or_mesh",
                    "quality": 0.45,
                })
            return summary

        main_face = max(faces, key=lambda f: self._bbox_area(f["bbox"]))
        bbox = main_face["bbox"]
        bbox_area = self._bbox_area(bbox)
        confidence = main_face.get("confidence", 0.0)
        if bbox_area < self.face_min_bbox_area:
            return summary

        quality = 0.25 + min(0.15, confidence * 0.15)
        mode = "bbox_only_possible_side_face"
        if landmarks is not None:
            quality += 0.20
            mode = "frontal_or_mesh"

        summary.update({
            "available": True,
            "main_face_bbox": bbox,
            "main_face_confidence": confidence,
            "mode": mode,
            "quality": min(1.0, quality),
        })
        return summary

    def _compute_presence(self, pose_summary: Dict, face_summary: Dict, now: float) -> Dict:
        sources = []
        score = 0.0
        if pose_summary.get("quality", 0) > 0:
            sources.append("pose")
            score += pose_summary["quality"] * 0.75
        if face_summary.get("available"):
            sources.append("face")
            score += face_summary["quality"] * 0.35
        if face_summary.get("landmarks_available"):
            sources.append("face_landmarks")
        score = min(1.0, score)
        self.presence_score_window.append(score)
        smoothed_score = sum(self.presence_score_window) / len(self.presence_score_window)
        recent_confirmed_ratio = sum(1 for v in self.presence_score_window if v >= self.presence_score_threshold) / len(self.presence_score_window)
        confirmed = smoothed_score >= self.presence_score_threshold or recent_confirmed_ratio >= self.presence_confirm_ratio
        if confirmed:
            self.last_confirmed_presence_time = now
        elif now - self.last_confirmed_presence_time <= self.presence_uncertain_grace_s and score > 0.25:
            confirmed = True

        if confirmed:
            reason = "confirmed_by_" + "+".join(sources) if sources else "confirmed_recently"
        elif score > 0:
            reason = "uncertain_" + "+".join(sources)
        else:
            reason = "no_person_evidence"

        return {
            "confirmed": confirmed,
            "score": score,
            "smoothed_score": smoothed_score,
            "source": sources,
            "reason": reason,
            "visible_landmarks": pose_summary.get("visible_landmarks", 0),
            "core_landmarks": pose_summary.get("core_landmarks", 0),
            "body_bbox": pose_summary.get("body_bbox"),
            "face_bbox": face_summary.get("main_face_bbox"),
        }

    def _bbox_distance_norm(self, a, b) -> float:
        if not a or not b:
            return 999.0
        ac = np.array([(a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0], dtype=np.float32)
        bc = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], dtype=np.float32)
        scale = max(1.0, float(np.hypot(a[2] - a[0], a[3] - a[1]) + np.hypot(b[2] - b[0], b[3] - b[1])) / 2.0)
        return float(np.linalg.norm(ac - bc) / scale)

    def _point_to_bbox_distance_norm(self, point, bbox) -> float:
        if point is None or not bbox:
            return 999.0
        px, py = point
        x1, y1, x2, y2 = bbox
        cx = min(max(px, x1), x2)
        cy = min(max(py, y1), y2)
        scale = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))
        return float(np.hypot(px - cx, py - cy) / scale)

    def _compute_baby_topology(self, pose_data: Optional[Dict], pose_summary: Dict, face_summary: Dict, face_pose: Dict, hands: List[Dict]) -> Dict:
        """Build a conservative baby topology anchored on face/head.

        This is not a new detector; it is a sanity layer to avoid trusting MediaPipe Pose when
        crib top-view, side-sleeping, blanket, or adult arms make head/limb assignment unstable.
        """
        head_bbox = face_summary.get("main_face_bbox") or pose_summary.get("head_bbox")
        pose_head_bbox = pose_summary.get("head_bbox")
        torso_bbox = pose_summary.get("torso_bbox")
        body_bbox = pose_summary.get("body_bbox")
        head_center = self._bbox_center(head_bbox) if head_bbox else pose_summary.get("head_center")
        torso_center = pose_summary.get("torso_center")
        body_center = pose_summary.get("body_center")

        face_anchor = bool(face_summary.get("landmarks_available") or face_summary.get("available"))
        head_pose_consistent = True
        if head_bbox and pose_head_bbox:
            head_pose_consistent = self._bbox_distance_norm(head_bbox, pose_head_bbox) < 1.25

        axis_confidence = 0.0
        axis_angle = None
        if head_center and (torso_center or body_center):
            other = torso_center or body_center
            dx = float(other[0] - head_center[0])
            dy = float(other[1] - head_center[1])
            axis_angle = float(np.degrees(np.arctan2(dy, dx)))
            axis_confidence += 0.35
        if face_anchor:
            axis_confidence += 0.30
        if pose_summary.get("core_landmarks", 0) >= self.pose_min_core_landmarks:
            axis_confidence += 0.20
        if torso_bbox:
            axis_confidence += 0.10
        if head_pose_consistent:
            axis_confidence += 0.05
        axis_confidence = min(1.0, axis_confidence)

        # External hands/arms: near enough to be possible occluders, but too far from exact airway
        # or not connected to baby topology; do not let them become baby limbs/exposure evidence.
        external_hands = []
        nearby_hands = []
        for hand in hands or []:
            hb = hand.get("bbox")
            d = self._bbox_distance_norm(hb, head_bbox) if hb and head_bbox else 999.0
            hand_info = dict(hand)
            hand_info["head_distance_norm"] = d
            if d < 2.2:
                nearby_hands.append(hand_info)
            external_hands.append(hand_info)

        posture = "unknown"
        if face_pose.get("available"):
            ori = face_pose.get("orientation")
            if ori == "front":
                posture = "supine_or_face_up"
            elif ori in ("slight_side", "side"):
                posture = "side_lying"
        elif head_bbox and torso_bbox:
            posture = "head_visible_pose_only"

        topology_reliable = axis_confidence >= 0.55 and head_pose_consistent
        return {
            "head_bbox": head_bbox,
            "head_center": head_center,
            "torso_bbox": torso_bbox,
            "body_bbox": body_bbox,
            "axis_angle": axis_angle,
            "axis_confidence": axis_confidence,
            "head_pose_consistent": head_pose_consistent,
            "topology_reliable": topology_reliable,
            "posture": posture,
            "nearby_hands": nearby_hands,
            "external_hands": external_hands,
            "external_hand_count": len(external_hands),
        }

    def _motion_point(self, pose_data: Optional[Dict], index: int, min_visibility: float = 0.45):
        if pose_data is None:
            return None
        landmarks = pose_data.get("landmarks")
        if landmarks is None or index >= len(landmarks):
            return None
        lm = landmarks[index]
        if lm[2] < min_visibility:
            return None
        return (float(lm[0]), float(lm[1]))

    def _point_distance(self, a, b) -> float:
        if a is None or b is None:
            return 0.0
        return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

    def _compute_motion_features(self, pose_data: Optional[Dict], pose_summary: Dict, now: float) -> Dict:
        """Estimate head and limb agitation from frame-to-frame pose displacement.

        Values are normalized by body size so they remain comparable across camera distance.
        Motion is a supporting cry signal, not enough by itself to declare crying.
        """
        body_bbox = pose_summary.get("body_bbox")
        if body_bbox:
            body_diag = max(30.0, float(np.hypot(body_bbox[2] - body_bbox[0], body_bbox[3] - body_bbox[1])))
        else:
            body_diag = 180.0

        current = {
            "time": now,
            "head": pose_summary.get("head_center"),
            "left_wrist": self._motion_point(pose_data, 15),
            "right_wrist": self._motion_point(pose_data, 16),
            "left_ankle": self._motion_point(pose_data, 27),
            "right_ankle": self._motion_point(pose_data, 28),
        }

        head_delta = 0.0
        limb_delta = 0.0
        visible_limb_count = 0
        if self.prev_motion_points:
            head_delta = self._point_distance(current.get("head"), self.prev_motion_points.get("head")) / body_diag
            limb_deltas = []
            for key in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
                if current.get(key) is not None and self.prev_motion_points.get(key) is not None:
                    limb_deltas.append(self._point_distance(current[key], self.prev_motion_points[key]) / body_diag)
            visible_limb_count = len(limb_deltas)
            limb_delta = float(sum(limb_deltas) / len(limb_deltas)) if limb_deltas else 0.0

        self.prev_motion_points = current
        self.head_motion_window.append(min(1.0, head_delta * 5.0))
        self.limb_motion_window.append(min(1.0, limb_delta * 4.0))
        head_motion = float(sum(self.head_motion_window) / len(self.head_motion_window)) if self.head_motion_window else 0.0
        limb_motion = float(sum(self.limb_motion_window) / len(self.limb_motion_window)) if self.limb_motion_window else 0.0
        agitation = min(1.0, 0.45 * head_motion + 0.55 * limb_motion)
        return {
            "head_motion": head_motion,
            "limb_motion": limb_motion,
            "agitation": agitation,
            "raw_head_delta": head_delta,
            "raw_limb_delta": limb_delta,
            "visible_limb_count": visible_limb_count,
        }

    def _combine_cry_evidence(self, expression_confidence: float, cry_features: Dict, face_pose: Dict, motion_features: Dict) -> Tuple[float, Dict]:
        orientation = face_pose.get("orientation") if face_pose else None
        yaw = face_pose.get("yaw_ratio") if face_pose else None
        abs_yaw = abs(float(yaw)) if yaw is not None else 1.0
        agitation = float(motion_features.get("agitation", 0.0))
        head_motion = float(motion_features.get("head_motion", 0.0))
        limb_motion = float(motion_features.get("limb_motion", 0.0))
        mouth_open_score = float(cry_features.get("mouth_open_score", 0.0))
        mouth_aspect_ratio = float(cry_features.get("mouth_aspect_ratio", 0.0))
        self.mouth_open_window.append(mouth_open_score)
        if self.prev_mouth_open_score is not None:
            self.mouth_variation_window.append(abs(mouth_open_score - self.prev_mouth_open_score))
        self.prev_mouth_open_score = mouth_open_score
        mouth_open_sustained = float(sum(self.mouth_open_window) / len(self.mouth_open_window)) if self.mouth_open_window else 0.0
        mouth_variation = float(sum(self.mouth_variation_window) / len(self.mouth_variation_window)) if self.mouth_variation_window else 0.0

        if orientation == "front" and abs_yaw < 0.18:
            reliability = "high"
            combined = 0.65 * expression_confidence + 0.25 * mouth_open_sustained + 0.10 * agitation
        elif orientation in ("front", "slight_side", "side") and abs_yaw < 0.65:
            # Side/crib view: do not ignore crying. A clearly open mouth sustained across frames
            # is meaningful, especially with any head/body agitation. Cap only when mouth is weak.
            reliability = "medium" if abs_yaw < 0.42 else "side_visual"
            combined = 0.42 * mouth_open_sustained + 0.28 * expression_confidence + 0.20 * agitation + 0.10 * min(1.0, mouth_variation * 3.0)
            if mouth_open_sustained >= 0.62 and mouth_aspect_ratio >= 0.42:
                combined = max(combined, 0.70 + min(0.15, agitation * 0.25))
            elif mouth_open_sustained >= 0.50 and (agitation >= 0.08 or head_motion >= 0.04 or limb_motion >= 0.04):
                combined = max(combined, 0.62)
            else:
                combined = min(combined, 0.58)
        else:
            reliability = "low"
            combined = 0.45 * mouth_open_sustained + 0.35 * agitation + 0.20 * min(expression_confidence, 0.45)
            combined = min(combined, 0.62)

        return min(1.0, float(combined)), {
            "face_orientation": orientation,
            "yaw_ratio": yaw,
            "motion_agitation": agitation,
            "head_motion": head_motion,
            "limb_motion": limb_motion,
            "mouth_open_score": mouth_open_score,
            "mouth_open_sustained": mouth_open_sustained,
            "mouth_open_variation": mouth_variation,
            "mouth_aspect_ratio": mouth_aspect_ratio,
            "expression_confidence_raw": float(expression_confidence),
            "cry_reliability": reliability,
        }

    def _compute_distress_evidence(self, presence: Dict, face_summary: Dict, cry_data: Dict, motion_features: Dict, occlusion_data: Optional[Dict]) -> Tuple[float, Dict]:
        """Visual distress fallback when cry audio/FaceMesh is unavailable.

        If baby is present, face/airway is hidden or risky, and there is head/limb agitation,
        treat it as suspected crying/distress so notifications do not only say occlusion.
        """
        if not presence.get("confirmed"):
            return 0.0, {"reason": "no_confirmed_presence"}
        cry_conf = float(cry_data.get("confidence", 0.0)) if cry_data else 0.0
        cry_status = cry_data.get("status") if cry_data else "missing"
        agitation = float(motion_features.get("agitation", 0.0))
        head_motion = float(motion_features.get("head_motion", 0.0))
        limb_motion = float(motion_features.get("limb_motion", 0.0))
        occlusion_conf = float((occlusion_data or {}).get("confidence", 0.0))
        occlusion_reason = (occlusion_data or {}).get("reason")
        face_landmarks = bool(face_summary.get("landmarks_available"))
        face_available = bool(face_summary.get("available"))

        hidden_or_risky = (not face_landmarks and face_available) or occlusion_conf >= 0.45
        airway_high_risk = occlusion_conf >= self.occlusion_threshold
        motion_distress = agitation >= 0.035 or head_motion >= 0.025 or limb_motion >= 0.045
        score = cry_conf
        if airway_high_risk and not face_landmarks:
            # In real baby monitoring, airway/face risk + unavailable mesh should not only notify
            # as occlusion. It is also suspected distress/cry because visual cry cannot be read.
            score = max(score, 0.68 + min(0.15, occlusion_conf * 0.15))
        elif hidden_or_risky:
            score = max(score, 0.45 + min(0.22, occlusion_conf * 0.30))
        if motion_distress:
            score += min(0.24, agitation * 1.6 + head_motion * 0.8 + limb_motion * 0.6)
        if cry_status in ("recent_hold", "suspected_no_mesh"):
            score = max(score, cry_conf)
        score = min(1.0, score)
        self.distress_window.append(score)
        smoothed = float(sum(self.distress_window) / len(self.distress_window)) if self.distress_window else score
        if cry_conf >= self.cry_threshold:
            reason = "visual_cry"
        elif airway_high_risk and not face_landmarks:
            reason = "airway_risk_cry_unreadable"
        elif hidden_or_risky and motion_distress:
            reason = "airway_or_face_hidden_with_motion"
        else:
            reason = "weak_distress_evidence"
        return smoothed, {
            "reason": reason,
            "cry_status": cry_status,
            "cry_confidence": cry_conf,
            "occlusion_confidence": occlusion_conf,
            "occlusion_reason": occlusion_reason,
            "motion_agitation": agitation,
            "head_motion": head_motion,
            "limb_motion": limb_motion,
            "face_landmarks_available": face_landmarks,
        }

    def process_frame(self, frame: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """处理一帧图像，返回检测结果和绘制后的帧"""
        self.frame_count += 1
        now = time.time()

        if self.frame_count % 10 == 0:
            self.fps = 10 / (now - self.last_frame_time)
            self.last_frame_time = now

        results = {
            "timestamp": now,
            "fps": self.fps,
            "events": [],
            "detections": {}
        }

        faces = self.face_detector.detect_faces(frame) if (self.cry_enabled or self.occlusion_enabled) else []
        hands = []
        pose_data = self.body_detector.detect_pose(frame) if (self.exposure_enabled or self.region_enabled) else None
        pose_summary = self._summarize_pose(pose_data, frame.shape)
        if self.occlusion_enabled and pose_summary.get("head_bbox"):
            hands = self.face_detector.detect_hands_near_head(frame, pose_summary.get("head_bbox"))
        landmarks = self.face_detector.detect_face_landmarks(frame, pose_summary.get("head_bbox")) if (self.cry_enabled or self.occlusion_enabled) else None
        face_pose = self.face_detector.classify_face_orientation(landmarks)
        face_summary = self._summarize_face(faces, landmarks)
        if face_pose.get("available"):
            face_summary["pose"] = face_pose
            # FaceMesh can succeed even when BlazeFace bbox detector returns n=0.
            # For UI/status, count this as one mesh-visible face.
            face_summary["available"] = True
            face_summary["face_count"] = max(1, int(face_summary.get("face_count", 0)))
            face_summary["main_face_confidence"] = max(float(face_summary.get("main_face_confidence", 0.0)), 1.0)
            if not face_summary.get("main_face_bbox"):
                face_summary["main_face_bbox"] = pose_summary.get("head_bbox")

        # MediaPipe FaceDetection/FaceMesh is mostly a frontal/near-frontal face detector.
        # In crib top-view scenes a side face can be obvious to humans while FaceMesh returns
        # nothing. If pose gives us a reliable head box, expose that as a side/head-visible
        # face summary for UI and presence logic, while still keeping landmarks unavailable
        # for features that truly need mouth/nose landmarks (cry/occlusion).
        if (not face_summary.get("available")) and pose_summary.get("head_bbox"):
            face_summary.update({
                "available": True,
                "main_face_bbox": pose_summary.get("head_bbox"),
                "main_face_confidence": float(pose_summary.get("quality", 0.0)),
                "mode": "pose_head_side_visible",
                "quality": max(float(face_summary.get("quality", 0.0)), 0.30),
                "pose": {"available": False, "orientation": "head_visible_mesh_unavailable"},
            })

        presence = self._compute_presence(pose_summary, face_summary, now)
        baby_topology = self._compute_baby_topology(pose_data, pose_summary, face_summary, face_pose, hands)

        results["detections"]["faces"] = faces
        results["detections"]["hands"] = hands
        results["detections"]["face_landmarks"] = landmarks is not None
        results["detections"]["face_summary"] = face_summary
        results["detections"]["face_orientation"] = face_pose
        results["detections"]["pose"] = pose_data is not None
        results["detections"]["pose_summary"] = pose_summary
        results["detections"]["presence"] = presence
        results["detections"]["baby_topology"] = baby_topology
        if pose_data is not None:
            results["pose_data"] = pose_data
        motion_features = self._compute_motion_features(pose_data, pose_summary, now)
        results["detections"]["motion"] = motion_features

        if self.face_absence_enabled:
            # Side face / partially visible face often fails MediaPipe FaceMesh,
            # but pose can still see head keypoints. Treat a pose-derived head box
            # as enough evidence that the head/face area is visible, otherwise
            # normal side-sleeping gets false "face not visible" alerts.
            head_bbox = pose_summary.get("head_bbox")
            head_visible_by_pose = bool(head_bbox) and pose_summary.get("core_landmarks", 0) >= 2
            face_visible = (
                face_summary.get("available", False)
                or face_summary.get("landmarks_available", False)
                or head_visible_by_pose
            )
            if presence["confirmed"] and not face_visible:
                if self.face_absence_start_time is None:
                    self.face_absence_start_time = now
                duration = now - self.face_absence_start_time
                results["detections"]["face_absence"] = {
                    "status": "not_visible",
                    "duration_s": duration,
                    "threshold_s": self.face_absence_duration_threshold,
                }
                if duration >= self.face_absence_duration_threshold and self._should_alert("face_not_visible"):
                    photo_path = self.storage.save_photo(frame, now)
                    details = {
                        "duration_s": float(duration),
                        "presence_score": float(presence.get("smoothed_score", 0.0)),
                        "reason": presence.get("reason"),
                    }
                    event_id = self.storage.save_event(
                        event_type="face_not_visible",
                        level="warning",
                        message=f"宝宝头脸区域连续不可见 {duration:.1f}s，请确认口鼻没有被遮挡",
                        details=details,
                        photo_path=photo_path
                    )
                    self.notifier.send_alert(
                        event_type="face_not_visible",
                        level="warning",
                        message="头脸区域不可见，请确认口鼻安全",
                        photo_path=photo_path,
                        details={"持续时间": f"{duration:.1f}s", "事件ID": event_id}
                    )
                    results["events"].append({
                        "type": "face_not_visible",
                        "level": "warning",
                        "duration_s": duration,
                        "event_id": event_id,
                        "photo_path": photo_path
                    })
            else:
                self.face_absence_start_time = None
                results["detections"]["face_absence"] = {
                    "status": "visible" if face_visible else "no_confirmed_person",
                    "duration_s": 0.0,
                    "threshold_s": self.face_absence_duration_threshold,
                }

        if self.cry_enabled:
            if landmarks is not None and presence["confirmed"]:
                expression_confidence, cry_features = self.face_detector.detect_cry_expression(landmarks)
                fused_cry, fusion_features = self._combine_cry_evidence(expression_confidence, cry_features, face_pose, motion_features)
                cry_features.update(fusion_features)
                smoothed_cry = self._get_smoothed_value(self.cry_confidence_window, fused_cry)
                results["detections"]["cry"] = {
                    "confidence": smoothed_cry,
                    "status": "available",
                    "features": cry_features
                }
                if smoothed_cry >= 0.55:
                    self.last_cry_confidence = smoothed_cry
                    self.last_cry_time = now
                if smoothed_cry >= self.cry_threshold:
                    if self.cry_start_time is None:
                        self.cry_start_time = now
                    elif now - self.cry_start_time >= self.cry_duration_threshold and self._should_alert("cry_detected"):
                        photo_path = self.storage.save_photo(frame, now)
                        level = "warning" if smoothed_cry < 0.85 else "danger"
                        event_id = self.storage.save_event(
                            event_type="cry_detected",
                            level=level,
                            message=f"检测到婴儿哭闹，融合置信度 {smoothed_cry:.2f}",
                            details=cry_features,
                            photo_path=photo_path
                        )
                        self.notifier.send_alert(
                            event_type="cry_detected",
                            level=level,
                            message="婴儿哭闹",
                            photo_path=photo_path,
                            details={
                                "融合置信度": f"{smoothed_cry:.2f}",
                                "表情原始分": f"{expression_confidence:.2f}",
                                "动作躁动": f"{motion_features.get('agitation', 0.0):.2f}",
                                "脸朝向": str(fusion_features.get("face_orientation")),
                                "事件ID": event_id,
                            }
                        )
                        results["events"].append({
                            "type": "cry_detected",
                            "level": level,
                            "confidence": smoothed_cry,
                            "event_id": event_id,
                            "photo_path": photo_path
                        })
                else:
                    self.cry_start_time = None
            else:
                self.cry_start_time = None
                fallback_mouth_score, fallback_mouth_features = self.face_detector.detect_mouth_open_fallback(frame, baby_topology.get("head_bbox") or pose_summary.get("head_bbox")) if presence["confirmed"] else (0.0, {})
                if presence["confirmed"] and fallback_mouth_score > 0:
                    self.mouth_open_window.append(fallback_mouth_score)
                    if self.prev_mouth_open_score is not None:
                        self.mouth_variation_window.append(abs(fallback_mouth_score - self.prev_mouth_open_score))
                    self.prev_mouth_open_score = fallback_mouth_score
                fallback_mouth_sustained = float(sum(self.mouth_open_window) / len(self.mouth_open_window)) if self.mouth_open_window else fallback_mouth_score
                recent_age = now - self.last_cry_time if self.last_cry_time else 999.0
                if presence["confirmed"] and fallback_mouth_sustained >= 0.50:
                    suspected = min(0.72, 0.45 + fallback_mouth_sustained * 0.35 + motion_features.get("agitation", 0.0) * 0.4)
                    results["detections"]["cry"] = {
                        "confidence": suspected,
                        "status": "suspected_mouth_no_mesh",
                        "reason": "mouth_open_fallback_without_face_mesh",
                        "features": {**fallback_mouth_features, "mouth_open_sustained": fallback_mouth_sustained, "motion": motion_features}
                    }
                    if suspected >= 0.55:
                        self.last_cry_confidence = suspected
                        self.last_cry_time = now
                elif presence["confirmed"] and recent_age <= self.cry_hold_s and self.last_cry_confidence >= 0.55:
                    results["detections"]["cry"] = {
                        "confidence": self.last_cry_confidence,
                        "status": "recent_hold",
                        "reason": "face_mesh_temporarily_unavailable_recent_cry",
                        "features": {"recent_age_s": recent_age, "motion": motion_features}
                    }
                elif presence["confirmed"] and (motion_features.get("agitation", 0.0) >= 0.18 or motion_features.get("head_motion", 0.0) >= 0.12):
                    suspected = min(0.62, 0.35 + motion_features.get("agitation", 0.0) * 0.9 + motion_features.get("head_motion", 0.0) * 0.4)
                    results["detections"]["cry"] = {
                        "confidence": suspected,
                        "status": "suspected_no_mesh",
                        "reason": "motion_distress_without_face_mesh",
                        "features": {"motion": motion_features}
                    }
                else:
                    results["detections"]["cry"] = {
                        "confidence": 0.0,
                        "status": "unavailable",
                        "reason": "face_landmarks_unavailable_or_presence_unconfirmed",
                        "features": {"motion": motion_features}
                    }

        if self.occlusion_enabled:
            if presence["confirmed"]:
                if landmarks is not None:
                    occlusion_confidence, occlusion_features = self.face_detector.detect_occlusion(frame, landmarks)
                    # With FaceMesh available, only an explicit hand/occluder overlapping the exact
                    # mouth-nose ROI is allowed to raise the score above the conservative visual cap.
                    hand_conf, hand_features = self.face_detector.score_hands_against_roi(
                        occlusion_features.get("roi_bbox"), hands
                    )
                    if hand_conf > occlusion_confidence:
                        occlusion_confidence = hand_conf
                        occlusion_features.update(hand_features)
                else:
                    occlusion_confidence, occlusion_features = self.face_detector.detect_head_occlusion_fallback(frame, pose_summary.get("head_bbox"), hands)

                smoothed_occlusion = self._get_smoothed_value(self.occlusion_confidence_window, occlusion_confidence)
                occlusion_status = "available" if landmarks is not None else "fallback"
                results["detections"]["occlusion"] = {
                    "confidence": smoothed_occlusion,
                    "status": occlusion_status,
                    "reason": occlusion_features.get("reason"),
                    "features": occlusion_features
                }
                if smoothed_occlusion >= self.occlusion_threshold:
                    if self.occlusion_start_time is None:
                        self.occlusion_start_time = now
                    elif now - self.occlusion_start_time >= self.occlusion_duration_threshold and self._should_alert("occlusion_detected"):
                        photo_path = self.storage.save_photo(frame, now)
                        event_id = self.storage.save_event(
                            event_type="occlusion_detected",
                            level="danger",
                            message=f"检测到口鼻/头脸遮挡风险，置信度 {smoothed_occlusion:.2f}",
                            details=occlusion_features,
                            photo_path=photo_path
                        )
                        self.notifier.send_alert(
                            event_type="occlusion_detected",
                            level="danger",
                            message="口鼻/头脸遮挡风险",
                            photo_path=photo_path,
                            details={"置信度": f"{smoothed_occlusion:.2f}", "原因": str(occlusion_features.get("reason")), "事件ID": event_id}
                        )
                        cry_data_now = results["detections"].get("cry", {})
                        # If face/airway is blocked and cry cannot be visually read, send a paired
                        # cry/distress notification too. This is what the parent expects when baby is
                        # audibly crying but FaceMesh is unavailable.
                        if presence.get("confirmed") and cry_data_now.get("status") in ("unavailable", "recent_hold", "suspected_no_mesh") and self._should_alert("cry_distress_paired"):
                            paired_id = self.storage.save_event(
                                event_type="cry_detected",
                                level="warning",
                                message=f"婴儿哭闹/烦躁疑似，同时存在口鼻/头脸遮挡风险 {smoothed_occlusion:.2f}",
                                details={"paired_occlusion_event_id": event_id, "occlusion": occlusion_features, "cry": cry_data_now},
                                photo_path=photo_path
                            )
                            self.notifier.send_alert(
                                event_type="cry_detected",
                                level="warning",
                                message="婴儿哭闹/烦躁疑似 + 遮挡风险",
                                photo_path=photo_path,
                                details={"遮挡置信度": f"{smoothed_occlusion:.2f}", "原因": str(occlusion_features.get("reason")), "事件ID": paired_id}
                            )
                            results["events"].append({
                                "type": "cry_detected",
                                "level": "warning",
                                "confidence": smoothed_occlusion,
                                "event_id": paired_id,
                                "photo_path": photo_path
                            })
                        results["events"].append({
                            "type": "occlusion_detected",
                            "level": "danger",
                            "confidence": smoothed_occlusion,
                            "event_id": event_id,
                            "photo_path": photo_path
                        })
                else:
                    self.occlusion_start_time = None
            else:
                self.occlusion_start_time = None
                results["detections"]["occlusion"] = {
                    "confidence": 0.0,
                    "status": "unavailable",
                    "reason": "presence_unconfirmed",
                    "features": {}
                }

        # Distress / suspected crying fallback after cry + occlusion are both known.
        if self.cry_enabled:
            cry_data_for_distress = results["detections"].get("cry", {})
            occlusion_data_for_distress = results["detections"].get("occlusion", {})
            distress_conf, distress_features = self._compute_distress_evidence(
                presence, face_summary, cry_data_for_distress, motion_features, occlusion_data_for_distress
            )
            results["detections"]["distress"] = {
                "confidence": distress_conf,
                "status": "available" if presence.get("confirmed") else "unavailable",
                "features": distress_features,
            }
            if distress_conf >= self.distress_threshold:
                if self.distress_start_time is None:
                    self.distress_start_time = now
                elif now - self.distress_start_time >= self.distress_duration_threshold and self._should_alert("cry_detected"):
                    photo_path = self.storage.save_photo(frame, now)
                    event_id = self.storage.save_event(
                        event_type="cry_detected",
                        level="warning",
                        message=f"检测到婴儿哭闹/烦躁疑似，融合置信度 {distress_conf:.2f}",
                        details=distress_features,
                        photo_path=photo_path
                    )
                    self.notifier.send_alert(
                        event_type="cry_detected",
                        level="warning",
                        message="婴儿哭闹/烦躁疑似",
                        photo_path=photo_path,
                        details={
                            "融合置信度": f"{distress_conf:.2f}",
                            "原因": str(distress_features.get("reason")),
                            "遮挡分": f"{distress_features.get('occlusion_confidence', 0.0):.2f}",
                            "动作": f"{distress_features.get('motion_agitation', 0.0):.2f}",
                            "事件ID": event_id,
                        }
                    )
                    results["events"].append({
                        "type": "cry_detected",
                        "level": "warning",
                        "confidence": distress_conf,
                        "event_id": event_id,
                        "photo_path": photo_path
                    })
            else:
                self.distress_start_time = None

        if self.exposure_enabled:
            if pose_data is not None and presence["confirmed"] and pose_summary["quality"] >= 0.35:
                exposure_ratio, exposure_features = self.body_detector.detect_limb_exposure(frame, pose_data)
                smoothed_exposure = self._get_smoothed_value(self.exposure_ratio_window, exposure_ratio)
                exposed_limbs = exposure_features.get("exposed_limbs", [])
                single_arm_only = len(exposed_limbs) == 1 and exposed_limbs[0] in ("left_arm", "right_arm")
                topology_reliable = baby_topology.get("topology_reliable", False)
                external_hand_count = baby_topology.get("external_hand_count", 0)
                coverage_status = "available"
                coverage_level = "normal"
                leg_or_body_exposed = any(limb in exposed_limbs for limb in ("left_leg", "right_leg", "torso"))
                if not topology_reliable:
                    coverage_status = "uncertain_topology"
                    coverage_level = "uncertain"
                elif external_hand_count and not leg_or_body_exposed:
                    # Adult hand/arm near the baby often pollutes skin/limb exposure. Keep it visible
                    # as a hint, but do not escalate to blanket/coverage warning.
                    coverage_status = "nearby_external_hand"
                    coverage_level = "nearby_hand"
                elif smoothed_exposure >= 0.55 and leg_or_body_exposed:
                    coverage_level = "body_or_legs_exposed"
                elif smoothed_exposure >= self.exposure_threshold:
                    coverage_level = "limb_exposed"
                exposure_features.update({
                    "coverage_status": coverage_status,
                    "coverage_level": coverage_level,
                    "topology_reliable": topology_reliable,
                    "external_hand_count": external_hand_count,
                    "posture": baby_topology.get("posture"),
                })
                results["detections"]["limb_exposure"] = {
                    "ratio": smoothed_exposure,
                    "status": coverage_status,
                    "features": exposure_features
                }
                effective_exposure_threshold = max(self.exposure_threshold, 0.45 if single_arm_only else self.exposure_threshold)
                allow_exposure_alert = topology_reliable and leg_or_body_exposed and not external_hand_count
                if allow_exposure_alert and smoothed_exposure >= effective_exposure_threshold:
                    if self.exposure_start_time is None:
                        self.exposure_start_time = now
                    elif now - self.exposure_start_time >= self.exposure_duration_threshold and self._should_alert("limb_exposure"):
                        photo_path = self.storage.save_photo(frame, now)
                        event_id = self.storage.save_event(
                            event_type="limb_exposure",
                            level="warning",
                            message=f"检测到肢体裸露（踢被子），裸露比例 {smoothed_exposure:.2f}",
                            details=exposure_features,
                            photo_path=photo_path
                        )
                        self.notifier.send_alert(
                            event_type="limb_exposure",
                            level="warning",
                            message="踢被子",
                            photo_path=photo_path,
                            details={"裸露比例": f"{smoothed_exposure:.2f}",
                                     "裸露肢体": ",".join(exposure_features.get("exposed_limbs", [])),
                                     "事件ID": event_id}
                        )
                        results["events"].append({
                            "type": "limb_exposure",
                            "level": "warning",
                            "ratio": smoothed_exposure,
                            "event_id": event_id,
                            "photo_path": photo_path
                        })
                else:
                    self.exposure_start_time = None
            else:
                self.exposure_start_time = None
                results["detections"]["limb_exposure"] = {
                    "ratio": 0.0,
                    "status": "unavailable",
                    "features": {}
                }

        if self.region_enabled:
            in_region = True
            region_status = "no_confirmed_person" if not presence["confirmed"] else "uncertain"
            decision_basis = presence["reason"]
            body_overlap = 0.0
            torso_overlap = 0.0
            body_center_in_region = False
            torso_center_in_region = False
            head_center_in_region = False

            if presence["confirmed"]:
                body_bbox = pose_summary.get("body_bbox")
                torso_bbox = pose_summary.get("torso_bbox")
                head_bbox = pose_summary.get("head_bbox") or face_summary.get("main_face_bbox")
                body_center = pose_summary.get("body_center")
                torso_center = pose_summary.get("torso_center")
                head_center = pose_summary.get("head_center") or self._bbox_center(face_summary.get("main_face_bbox"))

                if body_bbox:
                    body_overlap, _ = self.region_detector.bbox_overlap_with_region(body_bbox)
                if torso_bbox:
                    torso_overlap, _ = self.region_detector.bbox_overlap_with_region(torso_bbox)
                if body_center:
                    body_center_in_region = self.region_detector.point_in_region(body_center)
                if torso_center:
                    torso_center_in_region = self.region_detector.point_in_region(torso_center)
                if head_center:
                    head_center_in_region = self.region_detector.point_in_region(head_center)

                if torso_center_in_region:
                    region_status = "in_region"
                    decision_basis = "torso_center"
                elif torso_overlap >= self.region_torso_overlap_threshold:
                    region_status = "in_region"
                    decision_basis = "torso_overlap"
                elif head_center_in_region and body_overlap >= 0.20:
                    region_status = "in_region"
                    decision_basis = "head_center"
                elif body_center_in_region or body_overlap >= self.region_body_overlap_threshold:
                    region_status = "in_region"
                    decision_basis = "body_overlap"
                else:
                    in_region = False
                    region_status = "out_of_region"
                    decision_basis = "confirmed_person_outside_region"

            region_features = {
                "person_detected": presence["confirmed"],
                "presence_score": presence["smoothed_score"],
                "body_bbox": pose_summary.get("body_bbox"),
                "torso_bbox": pose_summary.get("torso_bbox"),
                "head_bbox": pose_summary.get("head_bbox") or face_summary.get("main_face_bbox"),
                "body_overlap_ratio": body_overlap,
                "torso_overlap_ratio": torso_overlap,
                "body_center_in_region": body_center_in_region,
                "torso_center_in_region": torso_center_in_region,
                "head_center_in_region": head_center_in_region,
                "decision_basis": decision_basis,
                "visible_landmarks": pose_summary.get("visible_landmarks", 0),
                "core_landmarks": pose_summary.get("core_landmarks", 0),
                "topology_reliable": baby_topology.get("topology_reliable"),
                "posture": baby_topology.get("posture"),
                "axis_angle": baby_topology.get("axis_angle"),
                "axis_confidence": baby_topology.get("axis_confidence"),
            }
            results["detections"]["region"] = {
                "in_region": in_region,
                "status": region_status,
                "features": region_features
            }

            self.region_exit_window.append(not in_region and presence["confirmed"])
            exit_ratio = sum(1 for v in self.region_exit_window if v) / len(self.region_exit_window)
            if not presence["confirmed"] or region_status in ("no_confirmed_person", "uncertain"):
                self.region_exit_start_time = None
            elif not in_region and exit_ratio >= self.region_exit_confirm_ratio:
                if self.region_exit_start_time is None:
                    self.region_exit_start_time = now
                elif now - self.region_exit_start_time >= self.region_exit_duration_threshold and self._should_alert("region_exit"):
                    photo_path = self.storage.save_photo(frame, now)
                    event_id = self.storage.save_event(
                        event_type="region_exit",
                        level="warning",
                        message="婴儿离开安全区域",
                        details=region_features,
                        photo_path=photo_path
                    )
                    self.notifier.send_alert(
                        event_type="region_exit",
                        level="warning",
                        message="离开安全区域",
                        photo_path=photo_path,
                        details={"重叠比例": f"{body_overlap:.2f}", "事件ID": event_id}
                    )
                    results["events"].append({
                        "type": "region_exit",
                        "level": "warning",
                        "event_id": event_id,
                        "photo_path": photo_path
                    })
            else:
                self.region_exit_start_time = None

        self.last_results = results
        return results, frame

    def get_status_summary(self) -> Dict:
        """获取运行状态摘要"""
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "last_events": self.last_results.get("events", []),
            "has_cry": self.cry_start_time is not None or self.distress_start_time is not None,
            "has_exposure": self.exposure_start_time is not None,
            "has_occlusion": self.occlusion_start_time is not None,
            "has_region_exit": self.region_exit_start_time is not None,
            "has_face_absence": self.face_absence_start_time is not None,
        }

    def close(self):
        """释放资源"""
        self.face_detector.close()
        self.body_detector.close()
