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

        # 状态跟踪
        self.cry_start_time: Optional[float] = None
        self.exposure_start_time: Optional[float] = None
        self.occlusion_start_time: Optional[float] = None
        self.region_exit_start_time: Optional[float] = None

        # 平滑窗口
        self.cry_confidence_window = deque(maxlen=10)
        self.exposure_ratio_window = deque(maxlen=10)
        self.occlusion_confidence_window = deque(maxlen=10)
        self.presence_score_window = deque(maxlen=detection_cfg.get("presence_window_size", 8))
        self.region_exit_window = deque(maxlen=detection_cfg.get("region_exit_window_size", 8))
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
        landmarks = self.face_detector.detect_face_landmarks(frame) if (self.cry_enabled or self.occlusion_enabled) else None
        pose_data = self.body_detector.detect_pose(frame) if (self.exposure_enabled or self.region_enabled) else None
        pose_summary = self._summarize_pose(pose_data, frame.shape)
        face_summary = self._summarize_face(faces, landmarks)
        presence = self._compute_presence(pose_summary, face_summary, now)

        results["detections"]["faces"] = faces
        results["detections"]["face_landmarks"] = landmarks is not None
        results["detections"]["face_summary"] = face_summary
        results["detections"]["pose"] = pose_data is not None
        results["detections"]["pose_summary"] = pose_summary
        results["detections"]["presence"] = presence
        if pose_data is not None:
            results["pose_data"] = pose_data

        if self.cry_enabled:
            if landmarks is not None and presence["confirmed"]:
                cry_confidence, cry_features = self.face_detector.detect_cry_expression(landmarks)
                smoothed_cry = self._get_smoothed_value(self.cry_confidence_window, cry_confidence)
                results["detections"]["cry"] = {
                    "confidence": smoothed_cry,
                    "status": "available",
                    "features": cry_features
                }
                if smoothed_cry >= self.cry_threshold:
                    if self.cry_start_time is None:
                        self.cry_start_time = now
                    elif now - self.cry_start_time >= self.cry_duration_threshold and self._should_alert("cry_detected"):
                        photo_path = self.storage.save_photo(frame, now)
                        level = "warning" if smoothed_cry < 0.85 else "danger"
                        event_id = self.storage.save_event(
                            event_type="cry_detected",
                            level=level,
                            message=f"检测到婴儿哭闹，置信度 {smoothed_cry:.2f}",
                            details=cry_features,
                            photo_path=photo_path
                        )
                        self.notifier.send_alert(
                            event_type="cry_detected",
                            level=level,
                            message="婴儿哭闹",
                            photo_path=photo_path,
                            details={"置信度": f"{smoothed_cry:.2f}", "事件ID": event_id}
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
                results["detections"]["cry"] = {
                    "confidence": 0.0,
                    "status": "unavailable",
                    "reason": "face_landmarks_unavailable_or_presence_unconfirmed",
                    "features": {}
                }

        if self.occlusion_enabled:
            if landmarks is not None and presence["confirmed"]:
                occlusion_confidence, occlusion_features = self.face_detector.detect_occlusion(frame, landmarks)
                smoothed_occlusion = self._get_smoothed_value(self.occlusion_confidence_window, occlusion_confidence)
                results["detections"]["occlusion"] = {
                    "confidence": smoothed_occlusion,
                    "status": "available",
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
                            message=f"检测到口鼻遮挡，置信度 {smoothed_occlusion:.2f}",
                            details=occlusion_features,
                            photo_path=photo_path
                        )
                        self.notifier.send_alert(
                            event_type="occlusion_detected",
                            level="danger",
                            message="口鼻遮挡",
                            photo_path=photo_path,
                            details={"置信度": f"{smoothed_occlusion:.2f}", "事件ID": event_id}
                        )
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
                    "reason": "face_landmarks_unavailable_or_presence_unconfirmed",
                    "features": {}
                }

        if self.exposure_enabled:
            if pose_data is not None and presence["confirmed"] and pose_summary["quality"] >= 0.35:
                exposure_ratio, exposure_features = self.body_detector.detect_limb_exposure(frame, pose_data)
                smoothed_exposure = self._get_smoothed_value(self.exposure_ratio_window, exposure_ratio)
                results["detections"]["limb_exposure"] = {
                    "ratio": smoothed_exposure,
                    "status": "available",
                    "features": exposure_features
                }
                if smoothed_exposure >= self.exposure_threshold:
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
            "has_cry": self.cry_start_time is not None,
            "has_exposure": self.exposure_start_time is not None,
            "has_occlusion": self.occlusion_start_time is not None,
            "has_region_exit": self.region_exit_start_time is not None
        }

    def close(self):
        """释放资源"""
        self.face_detector.close()
        self.body_detector.close()
