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
from src.audio_gateway import AudioGateway, multimodal_fusion
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

        # 趴睡检测
        self.prone_enabled = detection_cfg.get("prone_detection_enabled", True)
        self.prone_duration_threshold = detection_cfg.get("prone_duration_threshold", 5.0)
        self.face_mesh_absence_duration_threshold = detection_cfg.get("face_mesh_absence_duration_threshold", 8.0)
        self.prone_start_time: Optional[float] = None
        self.face_mesh_absence_start_time: Optional[float] = None

        # 状态跟踪
        self.cry_start_time: Optional[float] = None
        self.exposure_start_time: Optional[float] = None
        self.occlusion_start_time: Optional[float] = None
        self.region_exit_start_time: Optional[float] = None
        self.region_enter_start_time: Optional[float] = None
        self.face_absence_start_time: Optional[float] = None
        # Edge-triggered region alerts: only notify on ENTER / EXIT state transitions,
        # not continuously while staying inside or outside the region.
        self.last_in_region: Optional[bool] = None
        # 证据预缓存：状态刚变化时立即抓拍，防抖确认后用缓存的照片发送通知
        # 确保通知画面与事件发生瞬间完全同步
        self.region_enter_candidate_photo: Optional[np.ndarray] = None
        self.region_exit_candidate_photo: Optional[np.ndarray] = None
        self.last_cry_confidence: float = 0.0
        self.last_cry_time: float = 0.0
        self.cry_hold_s: float = detection_cfg.get("cry_hold_s", 8.0)
        self.distress_start_time: Optional[float] = None
        self.distress_threshold = detection_cfg.get("distress_confidence_threshold", 0.70)
        self.distress_duration_threshold = detection_cfg.get("distress_duration_threshold", 1.5)

        # 平滑窗口
        self.cry_confidence_window = deque(maxlen=10)
        self.exposure_ratio_window = deque(maxlen=10)
        # Reduced window from 10 → 4 frames (≈2 seconds at 2fps) to avoid
        # ghost alarms: a transient high score 5 seconds ago should not
        # keep triggering alerts long after the occluder (hand/blanket) has moved away.
        # Short window + 1-second duration threshold still filters single-frame noise.
        self.occlusion_confidence_window = deque(maxlen=4)
        self.presence_score_window = deque(maxlen=detection_cfg.get("presence_window_size", 8))
        self.region_exit_window = deque(maxlen=detection_cfg.get("region_exit_window_size", 8))
        self.head_motion_window = deque(maxlen=detection_cfg.get("motion_window_size", 12))
        self.limb_motion_window = deque(maxlen=detection_cfg.get("motion_window_size", 12))
        self.mouth_open_window = deque(maxlen=detection_cfg.get("mouth_open_window_size", 6))
        self.distress_window = deque(maxlen=detection_cfg.get("distress_window_size", 6))
        self.prev_mouth_open_score: Optional[float] = None
        self.mouth_variation_window = deque(maxlen=detection_cfg.get("mouth_variation_window_size", 6))
        self.cry_temporal_window = deque(maxlen=detection_cfg.get("cry_temporal_window_size", 10))
        self.mouth_pulse_window = deque(maxlen=detection_cfg.get("mouth_pulse_window_size", 10))
        self.prev_mouth_trend: Optional[int] = None
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

        # 🎙️ 智能音频网关（独立进程模式，永不阻塞主循环）
        audio_cfg = self.config.get("audio", {})
        self.audio_enabled = audio_cfg.get("cry_detection_enabled", False)
        self.audio_gateway = AudioGateway(
            sample_rate=audio_cfg.get("sample_rate", 16000),  # 优化为16kHz，更轻量
            device_id=audio_cfg.get("device_id", None)
        )
        if self.audio_enabled:
            self.audio_gateway.start()  # 非阻塞，立即返回！
        self.last_audio_cry_confidence = 0.0
        self.audio_visual_fusion_enabled = True

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

    def stop(self):
        """停止监控，清理资源"""
        if self.audio_gateway:
            self.audio_gateway.stop()

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
        pose_quality = pose_summary.get("quality", 0)
        face_quality = face_summary.get("quality", 0)

        # ============================================================
        # 【姿态与人脸独立确认机制】
        # 核心思想：正睡靠Pose，侧睡靠Face，两者只要一个够强就算有人
        # - Pose 质量 >= 0.4 → 姿态确认
        # - Face 质量 >= 0.5 → 人脸确认（侧睡场景）
        # 避免：侧睡时Pose弱被误判为无人，同时过滤无Face的纯Pose误检
        # ============================================================
        pose_confirmed = pose_quality >= 0.40
        face_confirmed = face_quality >= 0.50
        has_valid_confirmation = pose_confirmed or face_confirmed

        if has_valid_confirmation:
            # 只要有一个确认源，基础得分直接拉到确认水平
            score = 0.85
            if pose_confirmed:
                sources.append("pose")
                score += pose_quality * 0.10  # Pose质量作为微调
            if face_confirmed:
                sources.append("face")
                score += face_quality * 0.10  # Face质量作为微调
                if face_summary.get("landmarks_available"):
                    sources.append("face_landmarks")
        else:
            # 没有确认源时，用加权和作为弱信号（0~0.7之间）
            if pose_quality > 0:
                score += pose_quality * 0.60
                sources.append("pose_weak")
            if face_quality > 0:
                score += face_quality * 0.30
                sources.append("face_weak")

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

        Also detects Moro reflex (startle): symmetric bilateral arm abduction, fast, brief.
        When Moro is detected, limb agitation is attenuated to avoid false cry/exposure signals.
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
        moro_detected = False
        moro_confidence = 0.0

        if self.prev_motion_points:
            head_delta = self._point_distance(current.get("head"), self.prev_motion_points.get("head")) / body_diag
            limb_deltas = []
            for key in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
                if current.get(key) is not None and self.prev_motion_points.get(key) is not None:
                    limb_deltas.append(self._point_distance(current[key], self.prev_motion_points[key]) / body_diag)
            visible_limb_count = len(limb_deltas)
            limb_delta = float(sum(limb_deltas) / len(limb_deltas)) if limb_deltas else 0.0

            # 【惊跳反射(Moro Reflex)检测】
            # 特征：双臂镜像对称外展，快速短暂（0.5-1.5秒）
            lw = current.get("left_wrist")
            rw = current.get("right_wrist")
            plw = self.prev_motion_points.get("left_wrist")
            prw = self.prev_motion_points.get("right_wrist")
            if lw is not None and rw is not None and plw is not None and prw is not None:
                # 左右手腕移动向量
                vl = np.array(lw, dtype=np.float32) - np.array(plw, dtype=np.float32)
                vr = np.array(rw, dtype=np.float32) - np.array(prw, dtype=np.float32)
                vl_norm = float(np.linalg.norm(vl))
                vr_norm = float(np.linalg.norm(vr))
                # 归一化距离阈值：手腕移动需要足够快
                fast_enough = (vl_norm / body_diag) >= 0.03 and (vr_norm / body_diag) >= 0.03
                if fast_enough and vl_norm > 0 and vr_norm > 0:
                    # 计算两向量夹角的余弦值（-1=反向, 0=垂直, 1=同向）
                    cos_angle = float(np.dot(vl, vr) / (vl_norm * vr_norm))
                    # Moro反射：双臂向外展开 → 向量接近反向（cos < -0.5，即夹角>120度）
                    if cos_angle < -0.5:
                        moro_confidence = min(1.0, (abs(cos_angle) - 0.5) / 0.5)
                        moro_detected = True
                        # 衰减肢体运动：Moro是非自主反射，不应作为哭闹/躁动证据
                        limb_delta *= 0.3

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
            "moro_detected": moro_detected,
            "moro_confidence": moro_confidence,
        }

    def _combine_cry_evidence(self, expression_confidence: float, cry_features: Dict, face_pose: Dict, motion_features: Dict) -> Tuple[float, Dict]:
        """Fuse visual cry evidence using temporal baby-cry features.

        Key idea: crying is usually a short temporal pattern, not a single-frame expression.
        - mouth rhythm/open-close variation is the primary visual cue;
        - head swing and limb agitation are supporting cues;
        - static open mouth with low motion is capped to avoid sleep/open-mouth breathing false positives.
        """
        orientation = face_pose.get("orientation") if face_pose else None
        yaw = face_pose.get("yaw_ratio") if face_pose else None
        abs_yaw = abs(float(yaw)) if yaw is not None else 1.0
        agitation = float(motion_features.get("agitation", 0.0))
        head_motion = float(motion_features.get("head_motion", 0.0))
        limb_motion = float(motion_features.get("limb_motion", 0.0))
        mouth_open_score = float(cry_features.get("mouth_open_score", 0.0))
        mouth_aspect_ratio = float(cry_features.get("mouth_aspect_ratio", 0.0))

        # Mouth temporal features.
        self.mouth_open_window.append(mouth_open_score)
        mouth_delta = 0.0
        if self.prev_mouth_open_score is not None:
            mouth_delta = abs(mouth_open_score - self.prev_mouth_open_score)
            self.mouth_variation_window.append(mouth_delta)
            trend = 1 if mouth_open_score - self.prev_mouth_open_score > 0.04 else -1 if self.prev_mouth_open_score - mouth_open_score > 0.04 else 0
            if trend and self.prev_mouth_trend and trend != self.prev_mouth_trend:
                self.mouth_pulse_window.append(1.0)
            else:
                self.mouth_pulse_window.append(0.0)
            if trend:
                self.prev_mouth_trend = trend
        self.prev_mouth_open_score = mouth_open_score

        mouth_open_sustained = float(sum(self.mouth_open_window) / len(self.mouth_open_window)) if self.mouth_open_window else 0.0
        mouth_variation = float(sum(self.mouth_variation_window) / len(self.mouth_variation_window)) if self.mouth_variation_window else 0.0
        mouth_pulse_rate = float(sum(self.mouth_pulse_window) / len(self.mouth_pulse_window)) if self.mouth_pulse_window else 0.0

        mouth_static_open_score = max(0.0, min(1.0, (mouth_open_sustained - 0.45) / 0.35))
        mouth_rhythm_score = max(
            0.0,
            min(1.0, 0.65 * min(1.0, mouth_variation * 4.0) + 0.35 * min(1.0, mouth_pulse_rate * 3.0))
        )
        head_swing_score = max(0.0, min(1.0, head_motion * 5.0))
        limb_agitation_score = max(0.0, min(1.0, limb_motion * 3.2))
        motion_burst_score = max(0.0, min(1.0, agitation * 3.0))

        # Expression reliability by view angle.
        if orientation == "front" and abs_yaw < 0.18:
            reliability = "high"
            expression_weight = 0.25
        elif orientation in ("front", "slight_side", "side") and abs_yaw < 0.65:
            reliability = "medium" if abs_yaw < 0.42 else "side_visual"
            expression_weight = 0.18
        else:
            reliability = "low"
            expression_weight = 0.10

        cry_temporal_score = (
            0.38 * mouth_rhythm_score
            + 0.22 * mouth_static_open_score
            + 0.16 * head_swing_score
            + 0.16 * limb_agitation_score
            + 0.08 * motion_burst_score
        )
        combined = cry_temporal_score * (1.0 - expression_weight) + expression_confidence * expression_weight

        # Strong cry pattern: rhythmic mouth + supporting motion.
        strong_temporal_cry = (
            mouth_rhythm_score >= 0.45
            and mouth_open_sustained >= 0.58
            and (head_swing_score >= 0.18 or limb_agitation_score >= 0.22 or motion_burst_score >= 0.20)
        )
        # Static mouth open without motion is common during sleep/yawning; cap it.
        static_open_no_motion = (
            mouth_static_open_score >= 0.45
            and mouth_rhythm_score < 0.22
            and head_swing_score < 0.16
            and limb_agitation_score < 0.18
        )

        if strong_temporal_cry:
            combined = max(combined, 0.70 + min(0.12, motion_burst_score * 0.08 + mouth_rhythm_score * 0.08))
        elif static_open_no_motion:
            combined = min(combined, 0.52)
        else:
            # Moderate evidence should stay visible but generally below notification threshold.
            combined = min(combined, 0.64)

        self.cry_temporal_window.append(combined)
        smoothed_temporal = float(sum(self.cry_temporal_window) / len(self.cry_temporal_window)) if self.cry_temporal_window else combined

        return min(1.0, float(smoothed_temporal)), {
            "face_orientation": orientation,
            "yaw_ratio": yaw,
            "motion_agitation": agitation,
            "head_motion": head_motion,
            "limb_motion": limb_motion,
            "mouth_open_score": mouth_open_score,
            "mouth_open_sustained": mouth_open_sustained,
            "mouth_open_variation": mouth_variation,
            "mouth_pulse_rate": mouth_pulse_rate,
            "mouth_static_open_score": mouth_static_open_score,
            "mouth_rhythm_score": mouth_rhythm_score,
            "head_swing_score": head_swing_score,
            "limb_agitation_score": limb_agitation_score,
            "motion_burst_score": motion_burst_score,
            "cry_temporal_score": cry_temporal_score,
            "mouth_aspect_ratio": mouth_aspect_ratio,
            "expression_confidence_raw": float(expression_confidence),
            "cry_reliability": reliability,
            "strong_temporal_cry": strong_temporal_cry,
            "static_open_no_motion": static_open_no_motion,
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

        hidden_or_risky = (not face_landmarks and face_available) or occlusion_conf >= 0.55
        airway_high_risk = occlusion_conf >= self.occlusion_threshold
        motion_distress = agitation >= 0.12 or head_motion >= 0.08 or limb_motion >= 0.14
        score = cry_conf
        if airway_high_risk and not face_landmarks:
            # Airway/face risk + unavailable mesh may indicate distress, but do not let weak
            # visual evidence repeatedly spam cry notifications without motion or strong risk.
            score = max(score, 0.64 + min(0.12, occlusion_conf * 0.12))
        elif hidden_or_risky and motion_distress:
            score = max(score, 0.45 + min(0.18, occlusion_conf * 0.25))
        if hidden_or_risky and motion_distress:
            score += min(0.20, agitation * 1.2 + head_motion * 0.5 + limb_motion * 0.4)
        elif not hidden_or_risky and motion_distress and cry_conf >= 0.62:
            score += min(0.10, agitation * 0.4 + limb_motion * 0.2)
        if cry_status in ("recent_hold", "suspected_no_mesh", "suspected_mouth_no_mesh"):
            score = max(score, cry_conf)
        score = min(1.0, score)
        self.distress_window.append(score)
        smoothed = float(sum(self.distress_window) / len(self.distress_window)) if self.distress_window else score
        if cry_conf >= self.cry_threshold:
            reason = "visual_cry"
        elif airway_high_risk and not face_landmarks and motion_distress:
            reason = "airway_risk_cry_unreadable"
        elif hidden_or_risky and motion_distress:
            reason = "airway_or_face_hidden_with_motion"
        else:
            reason = "weak_distress_evidence"
            score = min(score, 0.49)
            smoothed = min(smoothed, 0.49)
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

        # ============================================================
        # 【趴睡检测】
        # 2-3月龄婴儿无法自主翻身，面部朝下有窒息风险。
        # 侧睡本身正常（家长可能刻意侧放），只有当头框可见但FaceMesh
        # 持续检测不到面部关键点时，才高度怀疑面部朝下/埋入床垫。
        # ============================================================
        if self.prone_enabled and presence["confirmed"]:
            posture = baby_topology.get("posture", "unknown")
            face_landmarks_ok = face_summary.get("landmarks_available", False)
            head_bbox_exists = bool(baby_topology.get("head_bbox") or pose_summary.get("head_bbox"))

            if head_bbox_exists and not face_landmarks_ok:
                if self.face_mesh_absence_start_time is None:
                    self.face_mesh_absence_start_time = now
                face_mesh_absence_dur = now - self.face_mesh_absence_start_time
            else:
                self.face_mesh_absence_start_time = None
                face_mesh_absence_dur = 0.0

            prone_suspected = False
            prone_reason = ""

            # 头框可见但FaceMesh持续不可用 → 高度怀疑面部朝下/埋入床垫
            if head_bbox_exists and not face_landmarks_ok and face_mesh_absence_dur >= self.face_mesh_absence_duration_threshold:
                prone_suspected = True
                prone_reason = f"head_visible_face_mesh_missing_{face_mesh_absence_dur:.0f}s"

            if prone_suspected:
                if self.prone_start_time is None:
                    self.prone_start_time = now
                prone_dur = now - self.prone_start_time
                results["detections"]["prone"] = {
                    "status": "suspected",
                    "duration_s": prone_dur,
                    "reason": prone_reason,
                    "posture": posture,
                }
                if prone_dur >= self.prone_duration_threshold and self._should_alert("prone_detected"):
                    photo_path = self.storage.save_photo(frame, now)
                    event_id = self.storage.save_event(
                        event_type="prone_detected",
                        level="danger",
                        message=f"疑似面部朝下（趴睡风险），面部不可见 {prone_dur:.1f}s",
                        details={"posture": posture, "reason": prone_reason, "face_mesh_absence_s": face_mesh_absence_dur},
                        photo_path=photo_path
                    )
                    self.notifier.send_alert(
                        event_type="prone_detected",
                        level="danger",
                        message="疑似面部朝下，请检查睡姿",
                        photo_path=photo_path,
                        details={"姿势": posture, "面部关键点不可读": f"{face_mesh_absence_dur:.0f}s", "事件ID": event_id}
                    )
                    results["events"].append({
                        "type": "prone_detected",
                        "level": "danger",
                        "duration_s": prone_dur,
                        "event_id": event_id,
                        "photo_path": photo_path
                    })
            else:
                self.prone_start_time = None
                results["detections"]["prone"] = {
                    "status": "normal",
                    "duration_s": 0.0,
                    "reason": f"posture_{posture}_face_landmarks_{'ok' if face_landmarks_ok else 'missing'}",
                }
        elif self.prone_enabled:
            self.prone_start_time = None
            self.face_mesh_absence_start_time = None

        # 🎙️ 获取最新音频特征（永远不阻塞！队列为空返回上次值）
        audio_features = self.audio_gateway.get_latest_features() if self.audio_enabled else None
        if audio_features:
            self.last_audio_cry_confidence = audio_features.cry_confidence
            results["detections"]["audio"] = {
                "volume": audio_features.rms_volume,
                "cry_confidence": audio_features.cry_confidence,
                "pattern_match": audio_features.cry_pattern_match,
                "is_crying": audio_features.is_crying,
                "latency_ms": audio_features.processing_latency_ms,
                "gateway_healthy": self.audio_gateway.is_healthy() if self.audio_enabled else False
            }

        if self.cry_enabled:
            if landmarks is not None and presence["confirmed"]:
                expression_confidence, cry_features = self.face_detector.detect_cry_expression(landmarks)
                fused_cry, fusion_features = self._combine_cry_evidence(expression_confidence, cry_features, face_pose, motion_features)
                cry_features.update(fusion_features)
                smoothed_cry = self._get_smoothed_value(self.cry_confidence_window, fused_cry)

                # 多模态融合：音频 + 视觉 + 动作 + 嘴巴状态（永远不阻塞！）
                if self.audio_enabled and self.audio_visual_fusion_enabled and audio_features:
                    motion_confidence = float(motion_features.get("agitation", 0.0))
                    mouth_open = float(cry_features.get("mouth_open_score", 0.0))

                    # 使用新的多模态融合（纯数学运算，<0.1ms）
                    fusion_info = multimodal_fusion(
                        audio_features,
                        smoothed_cry,
                        motion_confidence,
                        mouth_open
                    )
                    cry_features.update(fusion_info)
                    smoothed_cry = fusion_info['fused_confidence']

                results["detections"]["cry"] = {
                    "confidence": smoothed_cry,
                    "status": "available",
                    "features": cry_features,
                    "fused": self.audio_enabled and audio_features is not None
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
                if presence["confirmed"] and fallback_mouth_sustained >= 0.65 and (motion_features.get("agitation", 0.0) >= 0.08 or motion_features.get("head_motion", 0.0) >= 0.06 or motion_features.get("limb_motion", 0.0) >= 0.10):
                    suspected = min(0.72, 0.40 + fallback_mouth_sustained * 0.32 + motion_features.get("agitation", 0.0) * 0.35)
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
                # 【修复3】大幅提高仅凭动作触发哭闹的阈值
                # 原阈值：agitation≥0.18 OR head_motion≥0.12 → 正常活动很容易达到
                # 新阈值：必须同时满足多个高阈值条件，且置信度上限降低
                elif presence["confirmed"] and (motion_features.get("agitation", 0.0) >= 0.45 and motion_features.get("head_motion", 0.0) >= 0.25 and motion_features.get("limb_motion", 0.0) >= 0.35):
                    suspected = min(0.50, 0.30 + motion_features.get("agitation", 0.0) * 0.4)  # 置信度上限降到0.5，低于告警阈值0.7
                    results["detections"]["cry"] = {
                        "confidence": suspected,
                        "status": "suspected_no_mesh",
                        "reason": "high_motion_distress_without_face_mesh",
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
                if smoothed_occlusion >= self.occlusion_threshold and occlusion_confidence >= 0.5:
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
                        # 【修复2】移除遮挡时的配对哭闹通知
                        # 原逻辑：只要有遮挡且FaceMesh不可用，就额外发送哭闹通知 → 导致大量误报
                        # 修改后：只发送遮挡通知，不附带哭闹误报
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

        # 【修复4】禁用 distress fallback 通知（最严重的误报来源）
        # 原逻辑：只要有轻微动作 + 轻微遮挡/脸部不可见 → 触发哭闹通知
        # 修改后：仅保留 distress 作为内部状态显示，完全禁用通知功能
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
                "notification": "disabled_to_reduce_false_positives"
            }
            # 不再发送任何 distress 相关的通知
            self.distress_start_time = None

        if self.exposure_enabled:
            if pose_data is not None and presence["confirmed"] and pose_summary["quality"] >= 0.35:
                exposure_ratio, exposure_features = self.body_detector.detect_limb_exposure(frame, pose_data)

                # 【惊跳反射过滤】Moro反射时手臂突然外展会瞬时提高肤色裸露比例。
                # 检测到惊跳时衰减 exposure_ratio 并延长确认时间，避免误报。
                moro_active = motion_features.get("moro_detected", False)
                if moro_active:
                    exposure_ratio *= 0.35  # 大幅衰减，惊跳不是真踢被子
                    exposure_features["moro_suppressed"] = True

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
            in_region = False  # 默认为不在区域内，有人且满足条件才设为 True
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

                # 【修复1】收紧区域判定逻辑：要求身体的主要部分在安全区域内
                # 不再因为"只有头在区域内"或"只有躯干中心在区域内"就判定在区域内
                # 必须满足：躯干和身体都有足够的重叠比例
                if torso_overlap >= 0.65 and body_overlap >= 0.50:
                    in_region = True
                    region_status = "in_region"
                    decision_basis = "torso_and_body_overlap"
                elif torso_center_in_region and body_overlap >= 0.45:
                    in_region = True
                    region_status = "in_region"
                    decision_basis = "torso_center_plus_body"
                elif torso_overlap >= 0.50 and body_center_in_region:
                    in_region = True
                    region_status = "in_region"
                    decision_basis = "torso_overlap_plus_center"
                else:
                    in_region = False
                    region_status = "out_of_region"
                    decision_basis = "body_not_sufficiently_inside_region"

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
            confirmed_exit = (
                presence["confirmed"]
                and region_status == "out_of_region"
                and exit_ratio >= self.region_exit_confirm_ratio
            )
            confirmed_in = (
                presence["confirmed"]
                and region_status == "in_region"
                and exit_ratio <= 0.3
                and torso_overlap >= 0.3  # 进入区域必须有至少30%躯干在床内，防止只伸手进去误判
            )

            # ============================================================
            # 【边缘触发 + 证据预缓存】区域事件通知
            # 核心机制：
            #   1. 刚检测到状态变化时 → 立即抓拍并缓存（事件发生瞬间的证据）
            #   2. 防抖持续确认期间 → 持续验证状态稳定性
            #   3. 确认后发送通知 → 用预缓存的照片，不是当前帧
            # 效果：通知画面与事件发生瞬间 100% 同步，同时保留防抖能力
            # ============================================================

            # 状态不确定或无人：清空所有计时器、缓存和状态记忆
            if not presence["confirmed"] or region_status in ("no_confirmed_person", "uncertain"):
                self.region_exit_start_time = None
                self.region_exit_candidate_photo = None
                self.region_enter_start_time = None
                self.region_enter_candidate_photo = None
                self.last_in_region = None  # 关键：无人时重置区域状态记忆，防止残留旧状态

            # 可能离开区域：检测到离开迹象 → 立即抓拍缓存 → 防抖确认
            elif confirmed_exit:
                if self.region_exit_start_time is None:
                    # 第0帧：刚检测到离开迹象 → 立即抓拍作为证据
                    self.region_exit_start_time = now
                    self.region_exit_candidate_photo = frame.copy()
                elif now - self.region_exit_start_time >= self.region_exit_duration_threshold:
                    # 确认后：用缓存的照片（状态刚变化时的那一帧）发送通知
                    should_notify_exit = self.last_in_region is not False
                    if should_notify_exit and self.region_exit_candidate_photo is not None:
                        photo_path = self.storage.save_photo(self.region_exit_candidate_photo, now)
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
                    # 更新状态，清空缓存
                    self.last_in_region = False
                    self.region_exit_start_time = None
                    self.region_exit_candidate_photo = None

            # 可能进入区域：检测到进入迹象 → 立即抓拍缓存 → 防抖确认
            elif confirmed_in:
                if self.region_enter_start_time is None:
                    # 第0帧：刚检测到进入迹象 → 立即抓拍作为证据
                    self.region_enter_start_time = now
                    self.region_enter_candidate_photo = frame.copy()
                elif now - self.region_enter_start_time >= self.region_exit_duration_threshold:
                    # 确认后：用缓存的照片（状态刚变化时的那一帧）发送通知
                    if (self.last_in_region is None or self.last_in_region is False) \
                            and self.region_enter_candidate_photo is not None:
                        photo_path = self.storage.save_photo(self.region_enter_candidate_photo, now)
                        event_id = self.storage.save_event(
                            event_type="region_enter",
                            level="warning",
                            message="婴儿进入安全区域",
                            details=region_features,
                            photo_path=photo_path
                        )
                        self.notifier.send_alert(
                            event_type="region_enter",
                            level="warning",
                            message="回到安全区域",
                            photo_path=photo_path,
                            details={"重叠比例": f"{body_overlap:.2f}", "事件ID": event_id}
                        )
                        results["events"].append({
                            "type": "region_enter",
                            "level": "info",
                            "event_id": event_id,
                            "photo_path": photo_path
                        })
                    # 更新状态，清空缓存
                    self.last_in_region = True
                    self.region_enter_start_time = None
                    self.region_enter_candidate_photo = None
                    self.region_exit_start_time = None  # 进入后，离开计时器也重置

            # 中间状态：重置计时器但保留缓存（避免单帧抖动导致缓存丢失）
            else:
                self.region_exit_start_time = None
                self.region_enter_start_time = None

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
            "has_prone": self.prone_start_time is not None,
        }

    def close(self):
        """释放资源"""
        self.face_detector.close()
        self.body_detector.close()
