"""
预览渲染模块
负责在帧上绘制检测结果和UI元素
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.config import get_config
from src.vision.region_detector import RegionDetector


class PreviewRenderer:
    def __init__(self):
        config = get_config()
        preview_cfg = config.get("preview", {})

        self.window_name = preview_cfg.get("window_name", "Baby Sleep Supervisor")
        self.show_help = preview_cfg.get("show_help", True)
        self.show_detection_boxes = preview_cfg.get("show_detection_boxes", True)
        self.show_safe_region = preview_cfg.get("show_safe_region", True)
        self.show_statistics = preview_cfg.get("show_statistics", True)

        # 颜色定义
        self.COLORS = {
            "normal": (0, 255, 0),      # 绿色 - 正常
            "warning": (0, 255, 255),   # 黄色 - 警告
            "danger": (0, 0, 255),      # 红色 - 危险
            "text": (255, 255, 255),    # 白色 - 文字
            "text_bg": (0, 0, 0),       # 黑色 - 文字背景
            "region": (0, 255, 0),      # 绿色 - 安全区域
            "face": (255, 0, 0),        # 蓝色 - 人脸框
            "pose": (255, 255, 0),      # 青色 - 姿态关键点
        }

        # 字体设置
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE_SMALL = 0.4
        self.FONT_SCALE_NORMAL = 0.5
        self.FONT_SCALE_LARGE = 0.6
        self.FONT_THICKNESS = 1

        # 骨架连接定义
        self.POSE_CONNECTIONS = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
            (25, 27), (26, 28)
        ]

        # 事件显示缓存
        self.last_events: List[Dict] = []
        self.event_display_time = 5.0  # 事件显示5秒

    def draw_face_detections(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """绘制人脸检测结果"""
        if not self.show_detection_boxes or not faces:
            return frame

        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            confidence = face["confidence"]

            # 绘制人脸框
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLORS["face"], 2)

            # 绘制置信度
            text = f"Face: {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), self.FONT,
                        self.FONT_SCALE_SMALL, self.COLORS["face"], self.FONT_THICKNESS)

        return frame

    def draw_pose_landmarks(self, frame: np.ndarray, pose_data: Optional[Dict]) -> np.ndarray:
        """绘制姿态关键点"""
        if not self.show_detection_boxes or pose_data is None:
            return frame

        landmarks = pose_data["landmarks"]

        # 绘制骨架连接
        for connection in self.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if landmarks[start_idx][2] > 0.5 and landmarks[end_idx][2] > 0.5:
                start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                cv2.line(frame, start_point, end_point, self.COLORS["pose"], 2)

        # 绘制关键点
        for i, landmark in enumerate(landmarks):
            if landmark[2] > 0.5:  # 只绘制可见的关键点
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(frame, (x, y), 3, self.COLORS["pose"], -1)

        return frame

    def draw_detection_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """绘制检测结果"""
        detections = results.get("detections", {})

        # 绘制哭闹检测结果
        if "cry" in detections:
            cry_data = detections["cry"]
            confidence = cry_data["confidence"]
            color = self.COLORS["danger"] if confidence > 0.7 else self.COLORS["warning"] if confidence > 0.5 else self.COLORS["normal"]

            text = f"Cry: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), self.FONT,
                        self.FONT_SCALE_NORMAL, color, self.FONT_THICKNESS)

        # 绘制遮挡检测结果
        if "occlusion" in detections:
            occlusion_data = detections["occlusion"]
            confidence = occlusion_data["confidence"]
            color = self.COLORS["danger"] if confidence > 0.6 else self.COLORS["warning"] if confidence > 0.3 else self.COLORS["normal"]

            text = f"Occlusion: {confidence:.2f}"
            cv2.putText(frame, text, (10, 50), self.FONT,
                        self.FONT_SCALE_NORMAL, color, self.FONT_THICKNESS)

            # 绘制口鼻区域框
            if "roi_bbox" in occlusion_data["features"]:
                x1, y1, x2, y2 = occlusion_data["features"]["roi_bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # 绘制肢体裸露检测结果
        if "limb_exposure" in detections:
            exposure_data = detections["limb_exposure"]
            ratio = exposure_data["ratio"]
            color = self.COLORS["danger"] if ratio > 0.5 else self.COLORS["warning"] if ratio > 0.3 else self.COLORS["normal"]

            text = f"Exposure: {ratio:.2f}"
            cv2.putText(frame, text, (10, 70), self.FONT,
                        self.FONT_SCALE_NORMAL, color, self.FONT_THICKNESS)

            # 显示裸露肢体
            exposed_limbs = exposure_data["features"].get("exposed_limbs", [])
            if exposed_limbs:
                limbs_text = f"Limbs: {', '.join(exposed_limbs)}"
                cv2.putText(frame, limbs_text, (10, 90), self.FONT,
                            self.FONT_SCALE_SMALL, color, self.FONT_THICKNESS)

        # 绘制区域检测结果
        if "region" in detections:
            region_data = detections["region"]
            in_region = region_data["in_region"]
            color = self.COLORS["normal"] if in_region else self.COLORS["danger"]

            text = f"In Region: {'Yes' if in_region else 'No'}"
            cv2.putText(frame, text, (10, 110), self.FONT,
                        self.FONT_SCALE_NORMAL, color, self.FONT_THICKNESS)

            # 绘制人体边界框
            if "body_bbox" in region_data["features"]:
                x1, y1, x2, y2 = region_data["features"]["body_bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        return frame

    def draw_safe_region(self, frame: np.ndarray, region_detector: RegionDetector) -> np.ndarray:
        """绘制安全区域"""
        if not self.show_safe_region:
            return frame

        return region_detector.draw_region(frame, self.COLORS["region"])

    def draw_status_bar(self, frame: np.ndarray, status: Dict) -> np.ndarray:
        """绘制状态栏"""
        h, w = frame.shape[:2]

        # FPS
        fps = status.get("fps", 0.0)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 100, 30), self.FONT,
                    self.FONT_SCALE_NORMAL, self.COLORS["text"], self.FONT_THICKNESS)

        # 系统状态
        status_texts = []
        if status.get("has_cry", False):
            status_texts.append(("CRY", self.COLORS["danger"]))
        if status.get("has_exposure", False):
            status_texts.append(("EXPOSURE", self.COLORS["warning"]))
        if status.get("has_occlusion", False):
            status_texts.append(("OCCLUSION", self.COLORS["danger"]))
        if status.get("has_region_exit", False):
            status_texts.append(("REGION EXIT", self.COLORS["warning"]))

        if not status_texts:
            status_texts.append(("NORMAL", self.COLORS["normal"]))

        y_offset = 50
        for text, color in status_texts:
            cv2.putText(frame, text, (w - 150, y_offset), self.FONT,
                        self.FONT_SCALE_SMALL, color, self.FONT_THICKNESS)
            y_offset += 20

        return frame

    def draw_events(self, frame: np.ndarray, events: List[Dict], current_time: float) -> np.ndarray:
        """绘制最近的事件"""
        if not events:
            return frame

        h, w = frame.shape[:2]
        y_offset = h - 20

        # 清理过期事件
        self.last_events = [e for e in self.last_events if current_time - e["timestamp"] < self.event_display_time]

        # 添加新事件
        for event in events:
            event["timestamp"] = current_time
            self.last_events.append(event)

        # 最多显示3个事件
        display_events = self.last_events[-3:]

        for event in reversed(display_events):
            event_type = event["type"]
            level = event["level"]
            color = self.COLORS[level]

            if event_type == "cry_detected":
                text = f"🚨 哭闹检测 (置信度: {event.get('confidence', 0):.2f})"
            elif event_type == "occlusion_detected":
                text = f"🚨 口鼻遮挡 (置信度: {event.get('confidence', 0):.2f})"
            elif event_type == "limb_exposure":
                text = f"⚠️  踢被子 (裸露比例: {event.get('ratio', 0):.2f})"
            elif event_type == "region_exit":
                text = f"⚠️  离开安全区域"
            else:
                text = f"事件: {event_type}"

            # 绘制半透明背景
            text_size = cv2.getTextSize(text, self.FONT, self.FONT_SCALE_SMALL, self.FONT_THICKNESS)[0]
            cv2.rectangle(frame, (10, y_offset - text_size[1] - 5),
                         (10 + text_size[0] + 10, y_offset + 5),
                         (0, 0, 0), -1)

            # 绘制文字
            cv2.putText(frame, text, (15, y_offset), self.FONT,
                        self.FONT_SCALE_SMALL, color, self.FONT_THICKNESS)

            y_offset -= 30

        return frame

    def draw_help_text(self, frame: np.ndarray) -> np.ndarray:
        """绘制帮助文本"""
        if not self.show_help:
            return frame

        h, w = frame.shape[:2]
        help_texts = [
            "快捷键:",
            "q: 退出程序",
            "h: 显示/隐藏帮助",
            "d: 显示/隐藏检测框",
            "r: 显示/隐藏安全区域",
            "s: 显示/隐藏统计信息",
            "c: 校准安全区域"
        ]

        y_offset = h - (len(help_texts) * 20) - 10
        x_offset = w - 200

        for text in help_texts:
            cv2.putText(frame, text, (x_offset, y_offset), self.FONT,
                        self.FONT_SCALE_SMALL, self.COLORS["text"], self.FONT_THICKNESS)
            y_offset += 20

        return frame

    def render(self, frame: np.ndarray, results: Dict, status: Dict, region_detector: RegionDetector) -> np.ndarray:
        """渲染完整的预览帧"""
        # 先绘制区域
        frame = self.draw_safe_region(frame, region_detector)

        # 绘制检测框
        detections = results.get("detections", {})
        if "faces" in detections:
            frame = self.draw_face_detections(frame, detections["faces"])
        if "pose" in detections and detections["pose"] and "pose_data" in results:
            frame = self.draw_pose_landmarks(frame, results["pose_data"])

        # 绘制检测结果文字
        frame = self.draw_detection_results(frame, results)

        # 绘制状态栏
        frame = self.draw_status_bar(frame, status)

        # 绘制事件
        frame = self.draw_events(frame, results.get("events", []), results.get("timestamp", 0))

        # 绘制帮助
        frame = self.draw_help_text(frame)

        return frame

    def show(self, frame: np.ndarray) -> int:
        """显示帧并等待按键
        返回按键码
        """
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        """关闭窗口"""
        cv2.destroyAllWindows()
