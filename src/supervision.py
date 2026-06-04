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

        # 设置安全区域
        safe_region = detection_cfg.get("safe_region", [[50, 50], [590, 430]])
        self.region_detector.set_safe_region((tuple(safe_region[0]), tuple(safe_region[1])))

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

        # 状态跟踪
        self.cry_start_time: Optional[float] = None
        self.exposure_start_time: Optional[float] = None
        self.occlusion_start_time: Optional[float] = None
        self.region_exit_start_time: Optional[float] = None

        # 平滑窗口
        self.cry_confidence_window = deque(maxlen=10)
        self.exposure_ratio_window = deque(maxlen=10)
        self.occlusion_confidence_window = deque(maxlen=10)

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

    def process_frame(self, frame: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """处理一帧图像，返回检测结果和绘制后的帧"""
        self.frame_count += 1
        now = time.time()

        # 计算FPS
        if self.frame_count % 10 == 0:
            self.fps = 10 / (now - self.last_frame_time)
            self.last_frame_time = now

        results = {
            "timestamp": now,
            "fps": self.fps,
            "events": [],
            "detections": {}
        }

        # 1. 人脸检测
        if self.cry_enabled or self.occlusion_enabled:
            faces = self.face_detector.detect_faces(frame)
            results["detections"]["faces"] = faces

            if faces:
                # 取最大的人脸
                main_face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                landmarks = self.face_detector.detect_face_landmarks(frame)
                results["detections"]["face_landmarks"] = landmarks is not None

                if landmarks is not None:
                    # 哭闹检测
                    if self.cry_enabled:
                        cry_confidence, cry_features = self.face_detector.detect_cry_expression(landmarks)
                        smoothed_cry = self._get_smoothed_value(self.cry_confidence_window, cry_confidence)
                        results["detections"]["cry"] = {
                            "confidence": smoothed_cry,
                            "features": cry_features
                        }

                        # 哭闹事件判断
                        if smoothed_cry >= self.cry_threshold:
                            if self.cry_start_time is None:
                                self.cry_start_time = now
                            elif now - self.cry_start_time >= self.cry_duration_threshold:
                                if self._should_alert("cry_detected"):
                                    # 保存照片和事件
                                    photo_path = self.storage.save_photo(frame, now)
                                    event_id = self.storage.save_event(
                                        event_type="cry_detected",
                                        level="warning" if smoothed_cry < 0.85 else "danger",
                                        message=f"检测到婴儿哭闹，置信度 {smoothed_cry:.2f}",
                                        details=cry_features,
                                        photo_path=photo_path
                                    )
                                    # 发送通知
                                    self.notifier.send_alert(
                                        event_type="cry_detected",
                                        level="warning" if smoothed_cry < 0.85 else "danger",
                                        message=f"婴儿哭闹",
                                        photo_path=photo_path,
                                        details={"置信度": f"{smoothed_cry:.2f}", "事件ID": event_id}
                                    )
                                    results["events"].append({
                                        "type": "cry_detected",
                                        "level": "warning" if smoothed_cry < 0.85 else "danger",
                                        "confidence": smoothed_cry,
                                        "event_id": event_id,
                                        "photo_path": photo_path
                                    })
                        else:
                            self.cry_start_time = None

                    # 口鼻遮挡检测
                    if self.occlusion_enabled:
                        occlusion_confidence, occlusion_features = self.face_detector.detect_occlusion(frame, landmarks)
                        smoothed_occlusion = self._get_smoothed_value(self.occlusion_confidence_window, occlusion_confidence)
                        results["detections"]["occlusion"] = {
                            "confidence": smoothed_occlusion,
                            "features": occlusion_features
                        }

                        # 遮挡事件判断
                        if smoothed_occlusion >= self.occlusion_threshold:
                            if self.occlusion_start_time is None:
                                self.occlusion_start_time = now
                            elif now - self.occlusion_start_time >= self.occlusion_duration_threshold:
                                if self._should_alert("occlusion_detected"):
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
                                        message=f"口鼻遮挡",
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

        # 2. 人体姿态检测
        if self.exposure_enabled or self.region_enabled:
            pose_data = self.body_detector.detect_pose(frame)
            results["detections"]["pose"] = pose_data is not None

            if pose_data:
                # 肢体裸露/踢被子检测
                if self.exposure_enabled:
                    exposure_ratio, exposure_features = self.body_detector.detect_limb_exposure(frame, pose_data)
                    smoothed_exposure = self._get_smoothed_value(self.exposure_ratio_window, exposure_ratio)
                    results["detections"]["limb_exposure"] = {
                        "ratio": smoothed_exposure,
                        "features": exposure_features
                    }

                    # 踢被子事件判断
                    if smoothed_exposure >= self.exposure_threshold:
                        if self.exposure_start_time is None:
                            self.exposure_start_time = now
                        elif now - self.exposure_start_time >= self.exposure_duration_threshold:
                            if self._should_alert("limb_exposure"):
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
                                    message=f"踢被子",
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

                # 区域检测
                if self.region_enabled:
                    in_region, region_features = self.body_detector.is_in_region(
                        pose_data, self.region_detector.safe_region
                    )
                    results["detections"]["region"] = {
                        "in_region": in_region,
                        "features": region_features
                    }

                    # 离开区域事件判断
                    if not in_region:
                        if self.region_exit_start_time is None:
                            self.region_exit_start_time = now
                        elif now - self.region_exit_start_time >= self.region_exit_duration_threshold:
                            if self._should_alert("region_exit"):
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
                                    message=f"离开安全区域",
                                    photo_path=photo_path,
                                    details={"重叠比例": f"{region_features.get('overlap_ratio', 0):.2f}",
                                             "事件ID": event_id}
                                )
                                results["events"].append({
                                    "type": "region_exit",
                                    "level": "warning",
                                    "event_id": event_id,
                                    "photo_path": photo_path
                                })
                    else:
                        self.region_exit_start_time = None

        # 3. 运动检测
        # motion_detected, motion_features = self.region_detector.detect_motion_in_region(frame)
        # results["detections"]["motion"] = motion_features

        # 保存最近结果
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
