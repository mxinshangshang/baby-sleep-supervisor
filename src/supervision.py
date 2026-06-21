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
        self.cry_threshold = detection_cfg.get("cry_confidence_threshold", 0.5)
        self.cry_duration_threshold = detection_cfg.get("cry_duration_threshold", 1.5)  # 哭声持续时间阈值（秒）
        self.exposure_threshold = detection_cfg.get("exposure_threshold", 0.3)
        self.occlusion_threshold = detection_cfg.get("occlusion_threshold", 0.6)
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

        # 趴睡检测
        self.prone_enabled = detection_cfg.get("prone_detection_enabled", True)

        # 人脸不可见状态（兼容旧代码 + 新帧计数逻辑）
        self.face_absence_start_time: Optional[float] = None

        # =================================================================
        # 【统一帧防抖】所有视频监控项默认3帧确认，差异化修改直接改各变量值即可
        # =================================================================
        self.CONFIRM_FRAMES: int = 3  # 统一默认值，需要差异化只需修改下面某个变量

        # 各监控项独立确认帧数（都默认等于统一常量）
        self.occlusion_confirm_frames: int = self.CONFIRM_FRAMES    # 遮挡检测
        self.exposure_confirm_frames: int = self.CONFIRM_FRAMES     # 肢体裸露/踢被子
        self.prone_confirm_frames: int = self.CONFIRM_FRAMES        # 趴睡检测
        self.face_absence_confirm_frames: int = self.CONFIRM_FRAMES # 人脸不可见
        self.region_confirm_frames: int = self.CONFIRM_FRAMES        # 区域进入/离开

        # 各监控项独立帧计数器（纯视觉检测用帧计数）
        self.occlusion_frames: int = 0                               # 遮挡帧计数
        self.exposure_frames: int = 0                                # 肢体裸露帧计数
        self.prone_frames: int = 0                                   # 趴睡帧计数
        self.face_absence_frames: int = 0                            # 人脸不可见帧计数
        self.region_exit_frames: int = 0                             # 区域离开帧计数
        self.region_enter_frames: int = 0                            # 区域进入帧计数

        # 哭声检测用时间防抖（音频是独立流，用时间更科学）
        self.cry_start_time: Optional[float] = None

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

        # 手部检测缓存：FaceMesh可用时每1秒跑一次，不可用时每帧跑
        self.hands_cache: List[Dict] = []
        self.hands_cache_time = 0.0
        self.hands_detect_interval_s = 1.0

        # FaceDetection 降频：上一帧FaceMesh成功时跳过BlazeFace
        self._last_facemesh_ok = False
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
        self.audio_only_cry_threshold = 0.3  # 纯音频置信度阈值
        self.audio_only_cry_duration = 1.5  # 纯音频持续时间阈值（秒）

        # 纯音频开始时间（音频独立，用时间防抖）
        self.audio_only_cry_start_time: Optional[float] = None

        # 兼容旧代码：时间相关阈值和状态变量
        self.face_absence_duration_threshold = 15.0  # 人脸不可见告警阈值（秒）
        self.exposure_duration_threshold = 5.0  # 肢体裸露持续时间阈值（秒）
        self.prone_duration_threshold = 5.0  # 趴睡持续时间阈值（秒）
        self.exposure_start_time: Optional[float] = None  # 肢体裸露开始时间
        self.prone_start_time: Optional[float] = None  # 趴睡开始时间
        self.face_mesh_absence_start_time: Optional[float] = None  # 人脸关键点不可见开始时间

        # 运行状态
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0

        # 维测：各检测器最近一次错误（key: 检测器名, value: {"count": int, "last": str, "time": float}）
        self._detector_errors: Dict[str, Dict] = {}

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
        in_region = True  # 默认值，区域检测启用时会被覆盖

        if self.cry_enabled or self.occlusion_enabled:
            if self._last_facemesh_ok and now - self.face_detector.last_face_detect_time < self.face_detector.face_detect_interval_s:
                faces = self.face_detector.last_face_detect_result
            else:
                faces = self.face_detector.detect_faces(frame)
                self.face_detector.last_face_detect_result = faces
                self.face_detector.last_face_detect_time = now
        else:
            faces = []
        hands = []
        pose_data = self.body_detector.detect_pose(frame) if (self.exposure_enabled or self.region_enabled) else None
        pose_summary = self._summarize_pose(pose_data, frame.shape)
        if self.occlusion_enabled and pose_summary.get("head_bbox"):
            should_detect_hands = (
                not self._last_facemesh_ok  # FaceMesh不可用时每帧跑
                or now - self.hands_cache_time >= self.hands_detect_interval_s
            )
            if should_detect_hands:
                hands = self.face_detector.detect_hands_near_head(frame, pose_summary.get("head_bbox"))
                self.hands_cache = hands
                self.hands_cache_time = now
            else:
                hands = self.hands_cache
        landmarks = self.face_detector.detect_face_landmarks(frame, pose_summary.get("head_bbox")) if (self.cry_enabled or self.occlusion_enabled) else None
        self._last_facemesh_ok = landmarks is not None
        face_pose = self.face_detector.classify_face_orientation(landmarks)
        face_summary = self._summarize_face(faces, landmarks)
        if face_pose.get("available"):
            face_summary["pose"] = face_pose
            # FaceMesh can succeed even when BlazeFace bbox detector returns n=0.
            # For UI/status, count this as one mesh-visible face.
            face_summary["available"] = True
            face_summary["face_count"] = max(1, int(face_summary.get("face_count", 0)))
            face_summary["main_face_confidence"] = max(float(face_summary.get("main_face_confidence", 0.0)), 1.0)
            # FaceMesh成功时直接从landmarks计算精准bbox，替代BlazeFace/head_bbox
            x_min, y_min = landmarks.min(axis=0)
            x_max, y_max = landmarks.max(axis=0)
            face_summary["main_face_bbox"] = (x_min, y_min, x_max, y_max)

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

        # ============================================================
        # 【先定义 in_region 和 region_status 变量，避免后面使用时未定义】
        # 默认 True：检测块（面部缺失/趴睡/哭闹场景2/遮挡/踢被子）依赖 in_region，
        # 但它们都在区域检测（1387行）之前执行。区域检测会在后面计算真实值，
        # 用于纯音频通道（1662行）和下一帧。1帧滞后 @3fps ≈ 333ms 可忽略。
        # ============================================================
        in_region = True
        region_status = "in_region"
        decision_basis = presence.get("reason", "unknown")
        body_overlap = 0.0
        torso_overlap = 0.0
        body_center_in_region = False
        torso_center_in_region = False
        head_center_in_region = False
        face_center_in_region = False
        face_bbox = face_summary.get("main_face_bbox")

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

        # 🎙️ 获取最新音频特征（永远不阻塞！队列为空返回上次值）
        # 必须在所有告警逻辑之前获取，以便各告警 details 都能附带音频上下文
        audio_features = self.audio_gateway.get_latest_features() if self.audio_enabled else None
        if audio_features:
            self.last_audio_cry_confidence = audio_features.cry_confidence
            results["detections"]["audio"] = {
                "volume": audio_features.rms_volume,
                "cry_confidence": audio_features.cry_confidence,
                "acoustic_confidence": audio_features.acoustic_confidence,
                "rhythm_score": audio_features.rhythm_score,
                "burst_count": audio_features.burst_count,
                "pattern_match": audio_features.cry_pattern_match,
                "is_crying": audio_features.is_crying,
                "latency_ms": audio_features.processing_latency_ms,
                "gateway_healthy": self.audio_gateway.is_healthy() if self.audio_enabled else False,
                "audio_seq": audio_features.seq,
                "cpu_temp_c": self._read_cpu_temp(),
            }

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
            if presence["confirmed"] and in_region and not face_visible:
                # 【统一帧防抖】连续3帧确认
                self.face_absence_frames += 1
                results["detections"]["face_absence"] = {
                    "status": "not_visible",
                    "frames": self.face_absence_frames,
                }
                if self.face_absence_frames >= self.face_absence_confirm_frames and self._should_alert("face_not_visible"):
                    photo_path = self.storage.save_photo(frame, now)
                    details = {
                        "frames": self.face_absence_frames,
                        "presence_score": float(presence.get("smoothed_score", 0.0)),
                        "reason": presence.get("reason"),
                    }
                    event_id = self.storage.save_event(
                        event_type="face_not_visible",
                        level="warning",
                        message=f"宝宝头脸区域连续不可见 {self.face_absence_frames} 帧，请确认口鼻没有被遮挡",
                        details=details,
                        photo_path=photo_path
                    )
                    self.notifier.send_alert(
                        event_type="face_not_visible",
                        level="warning",
                        message="头脸区域不可见，请确认口鼻安全",
                        photo_path=photo_path,
                        details={
                            "确认帧数": f"{self.face_absence_frames}帧",
                            "事件ID": event_id,
                            **self._build_alert_context(audio_features),
                        }
                    )
                    results["events"].append({
                        "type": "face_not_visible",
                        "level": "warning",
                        "frames": self.face_absence_frames,
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
        if self.prone_enabled and presence["confirmed"] and in_region:
            posture = baby_topology.get("posture", "unknown")
            face_landmarks_ok = face_summary.get("landmarks_available", False)
            head_bbox_exists = bool(baby_topology.get("head_bbox") or pose_summary.get("head_bbox"))

            if head_bbox_exists and not face_landmarks_ok:
                self.face_mesh_absence_start_time = now  # 保留兼容
                self.face_absence_frames += 1  # 帧计数
            else:
                self.face_mesh_absence_start_time = None
                self.face_absence_frames = max(0, self.face_absence_frames - 1)  # 衰减

            prone_suspected = False
            prone_reason = ""

            # 头框可见但FaceMesh持续不可用 → 高度怀疑面部朝下/埋入床垫
            if head_bbox_exists and not face_landmarks_ok and self.face_absence_frames >= self.face_absence_confirm_frames:
                prone_suspected = True
                prone_reason = f"head_visible_face_mesh_missing_{self.face_absence_frames}frames"

            if prone_suspected:
                self.prone_frames += 1
                results["detections"]["prone"] = {
                    "status": "suspected",
                    "frames": self.prone_frames,
                    "reason": prone_reason,
                    "posture": posture,
                }
                # 【统一帧防抖】3帧确认后触发告警
                if self.prone_frames >= self.prone_confirm_frames and self._should_alert("prone_detected"):
                    photo_path = self.storage.save_photo(frame, now)
                    event_id = self.storage.save_event(
                        event_type="prone_detected",
                        level="danger",
                        message=f"疑似面部朝下（趴睡风险），已确认 {self.prone_frames} 帧",
                        details={"posture": posture, "reason": prone_reason, "face_mesh_absence_frames": self.face_absence_frames},
                        photo_path=photo_path
                    )
                    self.notifier.send_alert(
                        event_type="prone_detected",
                        level="danger",
                        message="疑似面部朝下，请检查睡姿",
                        photo_path=photo_path,
                        details={
                            "姿势": posture,
                            "确认帧数": f"{self.prone_frames}帧",
                            "事件ID": event_id,
                            **self._build_alert_context(audio_features),
                        }
                    )
                    results["events"].append({
                        "type": "prone_detected",
                        "level": "danger",
                        "frames": self.prone_frames,
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

        if self.cry_enabled:
            # 场景1：有人脸关键点 → 完整多模态融合（表情 + 动作 + 音频）
            # 注意：不依赖 presence["confirmed"]，只要有人脸关键点就可以用表情辅助判断
            if landmarks is not None:
                expression_confidence, cry_features = self.face_detector.detect_cry_expression(landmarks)
                fused_cry, fusion_features = self._combine_cry_evidence(expression_confidence, cry_features, face_pose, motion_features)
                cry_features.update(fusion_features)
                smoothed_cry = self._get_smoothed_value(self.cry_confidence_window, fused_cry)

                # 多模态融合：音频 + 视觉 + 动作 + 嘴巴状态
                if self.audio_enabled and self.audio_visual_fusion_enabled and audio_features:
                    motion_confidence = float(motion_features.get("agitation", 0.0))
                    mouth_open = float(cry_features.get("mouth_open_score", 0.0))
                    fusion_info = multimodal_fusion(audio_features, smoothed_cry, motion_confidence, mouth_open)
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
                        audio_ctx = self._build_alert_context(audio_features)
                        event_id = self.storage.save_event(
                            event_type="cry_detected",
                            level=level,
                            message=f"检测到婴儿哭闹，融合置信度 {smoothed_cry:.2f}",
                            details={**cry_features, **audio_ctx},
                            photo_path=photo_path
                        )
                        alert_details = {
                            "融合置信度": f"{smoothed_cry:.2f}",
                            "表情原始分": f"{expression_confidence:.2f}",
                            "动作躁动": f"{motion_features.get('agitation', 0.0):.2f}",
                            "脸朝向": str(fusion_features.get("face_orientation")),
                            "事件ID": event_id,
                            **audio_ctx
                        }
                        self.notifier.send_alert(
                            event_type="cry_detected",
                            level=level,
                            message="婴儿哭闹",
                            photo_path=photo_path,
                            details=alert_details
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
            elif in_region and self.audio_enabled and audio_features and audio_features.cry_confidence >= 0.15:
                # 场景2：in_region 但 landmarks 不可用 / presence 未确认（盖被子/侧脸场景）
                # 降级评估：音频 + 动作，不用人脸表情
                motion_confidence = float(motion_features.get("agitation", 0.0))
                limb_motion = float(motion_features.get("limb_motion", 0.0))

                # 音频为主，动作为辅的融合逻辑（简单有效，无需表情）
                base_confidence = audio_features.cry_confidence
                if motion_confidence >= 0.3 or limb_motion >= 0.25:
                    base_confidence = min(0.95, base_confidence + 0.15)  # 有动作加持，提高置信度

                smoothed_cry = self._get_smoothed_value(self.cry_confidence_window, base_confidence)

                results["detections"]["cry"] = {
                    "confidence": smoothed_cry,
                    "status": "audio_motion_only",
                    "reason": "in_region_but_landmarks_or_presence_unconfirmed",
                    "features": {
                        "audio_confidence": audio_features.cry_confidence,
                        "motion_confidence": motion_confidence,
                        "limb_motion": limb_motion,
                    },
                    "fused": True
                }
                if smoothed_cry >= self.cry_threshold:
                    self.last_cry_confidence = smoothed_cry
                    self.last_cry_time = now
                # 音频为主的降级模式，用时间防抖（音频独立）
                if smoothed_cry >= self.cry_threshold:
                    if self.cry_start_time is None:
                        self.cry_start_time = now
                    elif now - self.cry_start_time >= 1.5 and self._should_alert("cry_detected"):
                        photo_path = self.storage.save_photo(frame, now) if frame is not None else None
                        audio_ctx = self._build_alert_context(audio_features)
                        event_id = self.storage.save_event(
                            event_type="cry_detected",
                            level="warning",
                            message=f"麦克风检测到哭声（侧脸/盖被子模式），置信度 {smoothed_cry:.2f}",
                            details={"detection_mode": "audio_motion_only", **audio_ctx},
                            photo_path=photo_path
                        )
                        self.notifier.send_alert(
                            event_type="cry_detected",
                            level="warning",
                            message="婴儿哭声（侧脸/盖被子模式）",
                            photo_path=photo_path,
                            details={
                                "置信度": f"{smoothed_cry:.2f}",
                                "检测模式": "音频+动作（无人脸表情）",
                                "事件ID": event_id,
                                **audio_ctx
                            }
                        )
                        results["events"].append({
                            "type": "cry_detected",
                            "level": "warning",
                            "source": "audio_motion_fusion",
                            "confidence": smoothed_cry,
                            "event_id": event_id,
                        })
                else:
                    self.cry_start_time = None
            else:
                # 场景3：不在区，完全交给后面的纯音频快速通道处理
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
                elif presence["confirmed"] and (motion_features.get("agitation", 0.0) >= 0.45 and motion_features.get("head_motion", 0.0) >= 0.25 and motion_features.get("limb_motion", 0.0) >= 0.35):
                    suspected = min(0.50, 0.30 + motion_features.get("agitation", 0.0) * 0.4)
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
            if presence["confirmed"] and in_region:
                if landmarks is not None:
                    # 通用遮挡检测：检测任何物体（手/被子/玩具/枕头/衣物等）
                    occlusion_confidence, occlusion_features = self.face_detector.detect_occlusion(frame, landmarks, hands)
                else:
                    # landmarks 不可用（侧脸/遮挡严重）→ 降级用头部 ROI 检测
                    occlusion_confidence, occlusion_features = self.face_detector.detect_head_occlusion_fallback(frame, pose_summary.get("head_bbox"), hands)

                smoothed_occlusion = self._get_smoothed_value(self.occlusion_confidence_window, occlusion_confidence)
                occlusion_status = "available" if landmarks is not None else "fallback"
                results["detections"]["occlusion"] = {
                    "confidence": smoothed_occlusion,
                    "status": occlusion_status,
                    "reason": occlusion_features.get("reason"),
                    "features": occlusion_features
                }
                # 【统一帧防抖】连续3帧确认，允许1帧波动衰减
                if smoothed_occlusion >= self.occlusion_threshold and occlusion_confidence >= 0.5:
                    self.occlusion_frames += 1
                else:
                    # 帧计数衰减（不是立刻清零），允许中间1帧波动
                    self.occlusion_frames = max(0, self.occlusion_frames - 1)
                
                # 3帧确认后触发告警
                if self.occlusion_frames >= self.occlusion_confirm_frames and self._should_alert("occlusion_detected"):
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
                        details={
                            "置信度": f"{smoothed_occlusion:.2f}",
                            "原因": str(occlusion_features.get("reason")),
                            "事件ID": event_id,
                            **self._build_alert_context(audio_features),
                        }
                    )
                    results["events"].append({
                        "type": "occlusion_detected",
                        "level": "danger",
                        "confidence": smoothed_occlusion,
                        "event_id": event_id,
                        "photo_path": photo_path
                    })
            else:
                # presence未确认时直接重置
                self.occlusion_frames = 0
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
            # 【盖被子场景优化】：放宽检测条件，只要在区里且有足够检测信号就尝试检测
            # 原条件：pose_data is not None and presence["confirmed"] and in_region and pose_summary["quality"] >= 0.35
            # 新逻辑：支持三种模式
            #   1. 完整模式：pose质量高 → 全功能肢体检测
            #   2. 降级模式：有部分检测信号但质量不够高 → 简单肤色+bbox检测
            #   3. 动作模式：肢体大幅运动 → 可能是踢被子
            if in_region:
                pose_quality_ok = pose_data is not None and presence["confirmed"] and pose_summary.get("quality", 0) >= 0.35
                has_partial_detection = pose_data is not None and pose_summary.get("quality", 0) >= 0.15  # 有弱信号
                has_limb_motion = motion_features.get("limb_motion", 0) >= 0.3  # 肢体明显运动

                exposure_ratio = 0.0
                exposure_features = {}
                coverage_status = "unavailable"
                coverage_level = "normal"

                if pose_quality_ok:
                    # 模式1：完整模式 - 全功能肢体检测
                    exposure_ratio, exposure_features = self.body_detector.detect_limb_exposure(frame, pose_data)
                    coverage_status = "available"
                elif has_partial_detection or has_limb_motion:
                    # 模式2：降级模式 - 盖被子/侧脸场景，用简单检测
                    # 即使看不到完整姿态，只要有肤色裸露或肢体运动，就可能是踢被子
                    exposure_ratio, exposure_features = self.body_detector.detect_limb_exposure_simple(frame, baby_topology)
                    if has_limb_motion:
                        exposure_features["limb_motion_detected"] = True
                        exposure_features["limb_motion_score"] = motion_features.get("limb_motion", 0)
                    coverage_status = "degraded_mode_blanket_side_face"

                if coverage_status != "unavailable":
                    # 【惊跳反射过滤】Moro反射时手臂突然外展会瞬时提高肤色裸露比例
                    moro_active = motion_features.get("moro_detected", False)
                    if moro_active:
                        exposure_ratio *= 0.35  # 大幅衰减，惊跳不是真踢被子
                        exposure_features["moro_suppressed"] = True

                    smoothed_exposure = self._get_smoothed_value(self.exposure_ratio_window, exposure_ratio)
                    exposed_limbs = exposure_features.get("exposed_limbs", [])
                    single_arm_only = len(exposed_limbs) == 1 and exposed_limbs[0] in ("left_arm", "right_arm")
                    topology_reliable = baby_topology.get("topology_reliable", False)
                    external_hand_count = baby_topology.get("external_hand_count", 0)
                    leg_or_body_exposed = any(limb in exposed_limbs for limb in ("left_leg", "right_leg", "torso"))

                    if not topology_reliable and coverage_status == "available":
                        coverage_status = "uncertain_topology"
                        coverage_level = "uncertain"
                    elif external_hand_count and not leg_or_body_exposed:
                        # 大人的手在旁边，不告警
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
                        "detection_mode": "full" if pose_quality_ok else "degraded",
                    })
                    results["detections"]["limb_exposure"] = {
                        "ratio": smoothed_exposure,
                        "status": coverage_status,
                        "features": exposure_features
                    }

                    # 踢被子告警：【统一帧防抖】3帧确认
                    trigger_threshold = self.exposure_threshold if pose_quality_ok else 0.45
                    if smoothed_exposure >= trigger_threshold:
                        self.exposure_frames += 1
                    else:
                        # 帧计数衰减，允许波动
                        self.exposure_frames = max(0, self.exposure_frames - 1)
                    
                    if self.exposure_frames >= self.exposure_confirm_frames and self._should_alert("limb_exposure"):
                        photo_path = self.storage.save_photo(frame, now)
                        event_id = self.storage.save_event(
                            event_type="limb_exposure",
                            level="warning",
                            message=f"检测到肢体裸露/踢被子，裸露比例 {smoothed_exposure:.2f}",
                            details=exposure_features,
                            photo_path=photo_path
                        )
                        mode_text = "（盖被子/侧脸模式）" if coverage_status == "degraded_mode_blanket_side_face" else ""
                        self.notifier.send_alert(
                            event_type="limb_exposure",
                            level="warning",
                            message=f"婴儿肢体裸露/踢被子{mode_text}",
                            photo_path=photo_path,
                            details={
                                "裸露比例": f"{smoothed_exposure:.2f}",
                                "检测模式": coverage_status,
                                "事件ID": event_id,
                            }
                        )
                        results["events"].append({
                            "type": "limb_exposure",
                            "level": "warning",
                            "confidence": smoothed_exposure,
                                "event_id": event_id,
                                "photo_path": photo_path
                            })
                    else:
                        self.exposure_start_time = None
                else:
                    self.exposure_start_time = None
                    results["detections"]["limb_exposure"] = {
                        "ratio": 0.0,
                        "status": "no_detection_signal",
                        "features": {"note": "盖被子场景检测信号不足，持续观察中"}
                    }
            else:
                self.exposure_start_time = None

        if self.region_enabled:
            in_region = False  # 重新计算，覆盖默认值
            region_status = "out_of_region"
            decision_basis = presence["reason"]
            body_overlap = 0.0
            torso_overlap = 0.0
            body_center_in_region = False
            torso_center_in_region = False
            head_center_in_region = False

            # ============================================================
            # 【第一步：先计算所有检测到的部分和安全区的位置关系】
            # 注意：不管 presence.confirmed 是 True 还是 False，都要计算
            # 因为：就算只看到一只手/一个头，只要在安全区里，也应该算 in_region
            # ============================================================
            body_bbox = pose_summary.get("body_bbox")
            torso_bbox = pose_summary.get("torso_bbox")
            head_bbox = pose_summary.get("head_bbox") or face_summary.get("main_face_bbox")
            body_center = pose_summary.get("body_center")
            torso_center = pose_summary.get("torso_center")
            head_center = pose_summary.get("head_center") or self._bbox_center(face_summary.get("main_face_bbox"))
            face_center = self._bbox_center(face_bbox) if face_bbox else None

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
            if face_center:
                face_center_in_region = self.region_detector.point_in_region(face_center)

            # ============================================================
            # 【第二步：根据位置比例关系判断 in/out】
            # 核心原则：人脸是金标准 > 头 > 躯干 > 身体
            # 盖被子场景：即使看不到身体，只要看到人脸就算在区里
            # 无检测信号时：保持上一状态，防止抖动
            # ============================================================
            body_thr = self.region_body_overlap_threshold
            torso_thr = self.region_torso_overlap_threshold
            body_loose_thr = max(0.20, body_thr - 0.15)
            torso_loose_thr = max(0.20, torso_thr - 0.15)

            # 先判断有没有任何有效检测信号
            has_any_detection = (face_center is not None
                                or head_center is not None
                                or body_center is not None
                                or torso_center is not None
                                or body_overlap > 0.05)

            if not has_any_detection:
                # 没有任何有效检测信号 → 保持上一帧状态，防止抖动
                in_region = self.last_in_region if self.last_in_region is not None else False
                region_status = "in_region" if in_region else "out_of_region"
                decision_basis = "no_detection_signal_keep_previous_state"
            else:
                # 有检测信号，按优先级判断

                # ⭐ 优先级最高：人脸中心在区域内 → 金标准，直接判定在区域内
                # 盖被子场景：即使看不到身体，只要看到人脸就算在区里
                if face_center_in_region:
                    in_region = True
                    region_status = "in_region"
                    presence["confirmed"] = True  # 有脸就有人
                    decision_basis = "face_center_in_region_gold_standard"

                # 优先级2：有脸但没身体（盖被子）+ 人脸在区内
                elif face_bbox is not None and face_center_in_region:
                    in_region = True
                    region_status = "in_region"
                    presence["confirmed"] = True  # 有脸就有人
                    decision_basis = "face_only_under_blanket"

                # 优先级3：头框中心在区域内 + 至少有少量重叠 → 姿态辅助佐证
                elif head_center_in_region and (body_overlap >= 0.20 or torso_overlap >= 0.20):
                    in_region = True
                    region_status = "in_region"
                    decision_basis = "head_center_in_region_with_overlap"

                # 优先级4：标准的躯干+身体双重阈值判断
                elif torso_overlap >= torso_thr and body_overlap >= body_thr:
                    in_region = True
                    region_status = "in_region"
                    decision_basis = "torso_and_body_overlap"

                # 优先级5：躯干中心 + 宽松身体重叠
                elif torso_center_in_region and body_overlap >= body_loose_thr:
                    in_region = True
                    region_status = "in_region"
                    decision_basis = "torso_center_plus_body"

                # 优先级6：躯干重叠 + 身体中心在区域内
                elif torso_overlap >= torso_loose_thr and body_center_in_region:
                    in_region = True
                    region_status = "in_region"
                    decision_basis = "torso_overlap_plus_center"

                # 以上都不满足 → 判定不在区
                else:
                    in_region = False
                    region_status = "out_of_region"
                    decision_basis = "no_body_part_in_region"

            region_features = {
                "person_detected": presence["confirmed"],
                "presence_score": presence["smoothed_score"],
                "body_bbox": pose_summary.get("body_bbox"),
                "torso_bbox": pose_summary.get("torso_bbox"),
                "head_bbox": pose_summary.get("head_bbox") or face_summary.get("main_face_bbox"),
                "face_bbox": face_bbox,
                "body_overlap_ratio": body_overlap,
                "torso_overlap_ratio": torso_overlap,
                "body_center_in_region": body_center_in_region,
                "torso_center_in_region": torso_center_in_region,
                "head_center_in_region": head_center_in_region,
                "face_center_in_region": face_center_in_region,
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
                "features": region_features,
                "exit_pending": False,
                "exit_pending_s": 0.0,
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
                and exit_ratio <= 0.5  # 进入和离开门槛一致
                and body_overlap >= self.region_body_overlap_threshold  # 身体必须在区域内，防止空床误检
            )

            # ============================================================
            # 【边缘触发 + 证据预缓存】区域事件通知
            # 核心机制：
            #   1. 刚检测到状态变化时 → 立即抓拍并缓存（事件发生瞬间的证据）
            #   2. 防抖持续确认期间 → 持续验证状态稳定性
            #   3. 确认后发送通知 → 用预缓存的照片，不是当前帧
            # 效果：通知画面与事件发生瞬间 100% 同步，同时保留防抖能力
            # ============================================================

            # 不在区域内：只清空离开相关的计时器和缓存
            # 注意：不清空 enter 相关计时器，否则轻微抖动会导致永远无法完成进入确认
            if region_status == "out_of_region":
                self.region_exit_candidate_photo = None
                # 初始化：如果 last_in_region 还是 None，说明是系统启动后首次判定
                if self.last_in_region is None:
                    self.last_in_region = False

            # 可能离开区域：检测到离开迹象 → 帧计数防抖 → 确认后发通知
            elif confirmed_exit:
                if self.region_exit_frames == 0:
                    # 第0帧：刚检测到离开迹象 → 立即抓拍作为证据
                    self.region_exit_frames = 1
                    self.region_exit_candidate_photo = frame.copy()
                else:
                    self.region_exit_frames += 1  # 帧计数+1
                if self.region_exit_frames < self.region_confirm_frames:
                    # 防抖确认中：通知 UI 显示进度
                    results["detections"]["region"]["exit_pending"] = True
                    results["detections"]["region"]["exit_pending_s"] = self.region_exit_frames
                elif self.region_exit_frames >= self.region_confirm_frames:
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
                                details={
                                    "重叠比例": f"{body_overlap:.2f}",
                                    "事件ID": event_id,
                                    **self._build_alert_context(audio_features),
                                }
                            )
                            results["events"].append({
                                "type": "region_exit",
                                "level": "warning",
                                "event_id": event_id,
                                "photo_path": photo_path
                            })
                        # 更新状态，清空缓存
                        self.last_in_region = False
                        self.region_exit_frames = 0
                        self.region_exit_candidate_photo = None

            # 可能进入区域：检测到进入迹象 → 帧计数防抖 → 确认后发通知
            elif confirmed_in:
                if self.region_enter_frames == 0:
                    # 第0帧：刚检测到进入迹象 → 立即抓拍作为证据
                    self.region_enter_frames = 1
                    self.region_enter_candidate_photo = frame.copy()
                else:
                    self.region_enter_frames += 1  # 帧计数+1
                if self.region_enter_frames < self.region_confirm_frames:
                    # 防抖确认中
                    pass
                elif self.region_enter_frames >= self.region_confirm_frames:
                        # 确认后：用缓存的照片（状态刚变化时的那一帧）发送通知
                        if self.last_in_region is None:
                            # 启动后首次确认宝宝在安全区，只初始化状态，不发回区通知
                            self.last_in_region = True
                        elif self.last_in_region is False \
                                and self.region_enter_candidate_photo is not None:
                            # 只有之前明确在区域外，才发送 region_enter
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
                            self.last_in_region = True
                        # 更新状态，清空缓存
                        self.region_enter_frames = 0
                        self.region_enter_candidate_photo = None

            # 中间状态：重置计时器，避免单帧抖动导致误判
            else:
                self.region_exit_frames = 0
                self.region_enter_frames = 0

        # 🔄 不在区域内：统一重置所有视觉检测状态，避免历史遗留问题
        # 宝宝回到区域后所有检测从零开始，不会因为离开前的状态立即触发告警
        if not in_region:
            # 不在区域时：重置所有计数器
            self.occlusion_frames = 0
            self.exposure_frames = 0
            self.prone_frames = 0
            self.face_absence_frames = 0
            self.cry_start_time = None  # 哭声检测时间重置
            self.audio_only_cry_start_time = None  # 纯音频时间防抖
            self.distress_start_time = None

        # 🎙️ 纯音频快速通道：不在安全区时激活（音频独立，用时间防抖）
        if self.audio_enabled and audio_features and not in_region:
            # 更新 cry 状态，让调试能看到纯音频通道在工作
            results["detections"]["cry"] = {
                "confidence": audio_features.cry_confidence,
                "status": "audio_only",
                "features": {"audio_confidence": audio_features.cry_confidence},
                "fused": True
            }
            
            # 音频独立，用时间防抖：必须 is_crying 为 True + 满足阈值 + 持续时间
            if audio_features.is_crying and audio_features.cry_confidence >= self.audio_only_cry_threshold:
                if self.audio_only_cry_start_time is None:
                    self.audio_only_cry_start_time = now
                elif now - self.audio_only_cry_start_time >= self.audio_only_cry_duration and self._should_alert("cry_detected"):
                    event_id = self.storage.save_event(
                        event_type="cry_detected",
                        level="warning",
                        message=f"麦克风检测到高置信度哭声，音频置信度 {audio_features.cry_confidence:.2f}",
                        details={"audio_confidence": audio_features.cry_confidence,
                                 "pattern_match": audio_features.cry_pattern_match,
                                 "volume": audio_features.rms_volume},
                        photo_path=None  # 离区不发图片
                    )
                    self.notifier.send_alert(
                        event_type="cry_detected",
                        level="warning",
                        message="麦克风检测到婴儿哭声（离区音频通道）",
                        photo_path=None,  # 离区不发图片
                        details={
                            "音频置信度": f"{audio_features.cry_confidence:.2f}",
                            "事件ID": event_id,
                            **self._build_alert_context(audio_features),
                        }
                    )
                    results["events"].append({
                        "type": "cry_detected",
                        "level": "warning",
                        "source": "audio_only",
                        "confidence": audio_features.cry_confidence,
                        "event_id": event_id,
                    })
            else:
                self.audio_only_cry_start_time = None
        else:
            self.audio_only_cry_start_time = None

        self.last_results = results

        # 🔍 全系统诊断快照：每 ~10 秒写入 events_debug，事后可还原任意时刻系统全貌
        if self.frame_count % 30 == 0 and self.storage:
            self.storage.save_diagnostic_snapshot(self._build_diagnostic_snapshot(results, audio_features))

        # 🔄 音频网关健康监控：不健康时自动重启
        if self.audio_enabled and not self.audio_gateway.is_healthy():
            self.storage.save_debug_event(
                event_type="audio_gateway_restart",
                level="warning",
                message="音频网关不健康，自动重启",
                details={"status": self.audio_gateway.get_health_status()},
            )
            self.audio_gateway.stop()
            self.audio_gateway.start()

        return results, frame

    @staticmethod
    def _read_cpu_temp() -> Optional[float]:
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return round(float(f.read().strip()) / 1000.0, 1)
        except Exception:
            return None

    def _track_error(self, detector: str, error_msg: str):
        """记录检测器错误，供诊断快照使用。"""
        now = time.time()
        entry = self._detector_errors.get(detector, {"count": 0, "last": "", "time": 0.0})
        entry["count"] += 1
        entry["last"] = str(error_msg)[:200]
        entry["time"] = now
        self._detector_errors[detector] = entry

    def _build_diagnostic_snapshot(self, results: Dict, audio_features) -> Dict:
        """构建全系统诊断快照，用于写入 events_debug 表。

        每条记录包含所有检测器的当前状态，事后通过 OpenClaw 查询即可
        还原任意时刻系统全貌，无需同时查多个表做 join。
        """
        dets = results.get("detections", {})
        presence = dets.get("presence", {})
        audio_det = dets.get("audio", {})
        cry = dets.get("cry", {})
        occlusion = dets.get("occlusion", {})
        exposure = dets.get("limb_exposure", {})
        region = dets.get("region", {})
        prone = dets.get("prone", {})
        face_absence = dets.get("face_absence", {})

        return {
            "frame": self.frame_count,
            "fps": round(self.fps, 1),
            "presence": {
                "confirmed": presence.get("confirmed", False),
                "score": round(presence.get("smoothed_score", 0), 3),
                "source": presence.get("source", []),
            },
            "audio": {
                "volume": round(audio_det.get("volume", 0), 4),
                "cry_conf": round(audio_det.get("cry_confidence", 0), 3),
                "acoustic": round(audio_det.get("acoustic_confidence", 0), 3),
                "rhythm": round(audio_det.get("rhythm_score", 0), 3),
                "bursts": audio_det.get("burst_count", 0),
                "is_crying": audio_det.get("is_crying", False),
                "healthy": audio_det.get("gateway_healthy", False),
                "seq": audio_det.get("audio_seq", 0),
            },
            "system": {
                "cpu_temp_c": self._read_cpu_temp(),
                "inference_fps": round(self.fps, 1),
            },
            "errors": dict(self._detector_errors),
            "cry": {
                "fused": round(cry.get("confidence", 0), 3),
                "status": cry.get("status", "unavailable"),
                "expression": round(cry.get("features", {}).get("expression_confidence_raw", 0), 3) if isinstance(cry.get("features"), dict) else 0,
            },
            "occlusion": {
                "conf": round(occlusion.get("confidence", 0), 3),
                "frames": self.occlusion_frames,
                "status": occlusion.get("status", "unavailable"),
            },
            "exposure": {
                "ratio": round(exposure.get("ratio", 0), 3),
                "level": exposure.get("features", {}).get("coverage_level", "normal") if isinstance(exposure.get("features"), dict) else "normal",
            },
            "region": {
                "in_region": region.get("in_region"),
                "status": region.get("status", "disabled"),
                "features": region.get("features", {}),
                "exit_pending": region.get("exit_pending", False),
                "exit_pending_frames": region.get("exit_pending_s", 0),
            },
            "prone": {
                "status": prone.get("status", "disabled"),
                "duration_s": round(prone.get("duration_s", 0), 1),
            },
            "face_absence": {
                "status": face_absence.get("status", "visible"),
                "duration_s": round(face_absence.get("duration_s", 0), 1),
            },
            "alerts": {
                "cry": self.cry_start_time is not None,
                "occlusion": self.occlusion_frames > 0,
                "exposure": self.exposure_start_time is not None,
                "region_exit": self.region_exit_frames > 0,
                "prone": self.prone_start_time is not None,
                "face_absence": self.face_absence_start_time is not None,
            },
        }

    def _build_alert_context(self, audio_features) -> Dict:
        """构建告警时的音频/系统上下文，统一附加到所有飞书告警 details 中。"""
        if not audio_features:
            return {}
        return {
            "音频综合分": f"{audio_features.cry_confidence:.2f}",
            "声学分": f"{audio_features.acoustic_confidence:.2f}",
            "节律分": f"{audio_features.rhythm_score:.2f}",
            "音量": f"{audio_features.rms_volume:.3f}",
            "连续爆发": f"{audio_features.burst_count}次",
            "是否检测到哭声": "是" if audio_features.is_crying else "否",
        }

    def get_status_summary(self) -> Dict:
        """获取运行状态摘要"""
        return {
            "fps": self.fps,
            "frame_count": self.frame_count,
            "last_events": self.last_results.get("events", []),
            "has_cry": self.cry_start_time is not None or self.distress_start_time is not None,
            "has_exposure": self.exposure_start_time is not None,
            "has_occlusion": self.occlusion_frames > 0,
            "has_region_exit": self.region_exit_frames > 0,
            "has_face_absence": self.face_absence_start_time is not None,
            "has_prone": self.prone_start_time is not None,
        }

    def close(self):
        """释放资源"""
        self.face_detector.close()
        self.body_detector.close()
