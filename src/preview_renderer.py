"""
Preview Renderer
Draw detection results and UI elements on frame
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
            "normal": (0, 255, 0),      # Green - Normal
            "warning": (0, 255, 255),   # Yellow - Warning
            "danger": (0, 0, 255),      # Red - Danger
            "info": (0, 255, 0),        # Green - Info (same as normal, for region enter)
            "text": (255, 255, 255),    # White - Text
            "text_bg": (0, 0, 0),       # Black - Text background
            "region": (0, 255, 0),      # Green - Safe region
            "face": (255, 0, 0),        # Blue - Face box
            "pose": (255, 255, 0),      # Cyan - Pose keypoints
            "torso": (255, 0, 255),     # Purple - Torso box
            "head": (0, 165, 255),      # Orange - Head box
            "nearby": (0, 255, 255),    # Yellow - Nearby hand/object
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

    def draw_hand_detections(self, frame: np.ndarray, hands: List[Dict], occlusion: Optional[Dict] = None) -> np.ndarray:
        """Draw hands/arms with risk semantics: nearby != airway occluder."""
        if not self.show_detection_boxes or not hands:
            return frame
        risky_boxes = []
        if occlusion:
            for h in occlusion.get("features", {}).get("overlapping_hands", []) or []:
                if h.get("bbox"):
                    risky_boxes.append(tuple(h["bbox"]))
        for hand in hands:
            x1, y1, x2, y2 = hand["bbox"]
            is_risky = any(abs(x1-r[0]) < 5 and abs(y1-r[1]) < 5 and abs(x2-r[2]) < 5 and abs(y2-r[3]) < 5 for r in risky_boxes)
            color = self.COLORS["danger"] if is_risky else self.COLORS["nearby"]
            label = "Airway occluder" if is_risky else "Hand nearby"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(12, y1 - 8)), self.FONT,
                        self.FONT_SCALE_SMALL, color, self.FONT_THICKNESS)
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
        """绘制检测结果 — 固定双列布局。

        左列：安全状态（Presence / Face / Region / Occlusion / Exposure / Cry）
        右列：传感器诊断（FPS / Audio / Temp / Motion）
        底部：活跃告警（face_absence / prone / distress）
        """
        detections = results.get("detections", {})
        h, w = frame.shape[:2]
        ROW_H = 18
        LEFT_X = 10
        RIGHT_X = w - 260
        Y = 28

        # ═══════════════════════════════════════
        # 左列：安全状态
        # ═══════════════════════════════════════

        # 1. Presence
        presence = detections.get("presence", {})
        confirmed = presence.get("confirmed", False)
        pscore = presence.get("smoothed_score", 0.0)
        if confirmed:
            ptext, pcolor = "Presence: OK", self.COLORS["normal"]
        elif pscore > 0.2:
            ptext, pcolor = f"Presence: uncertain {pscore:.2f}", self.COLORS["warning"]
        else:
            ptext, pcolor = "Presence: none", self.COLORS["danger"]
        cv2.putText(frame, ptext, (LEFT_X, Y), self.FONT, self.FONT_SCALE_SMALL, pcolor, self.FONT_THICKNESS)

        # 2. Face
        face_summary = detections.get("face_summary", {})
        face_mode = face_summary.get("mode", "not_visible")
        face_pose = face_summary.get("pose", {}) or detections.get("face_orientation", {})
        orientation = face_pose.get("orientation")
        if face_mode == "frontal_or_mesh":
            if orientation == "front":
                ftext = "Face: front mesh"
            elif orientation in ("slight_side", "side"):
                ftext = f"Face: {orientation} mesh"
            else:
                ftext = "Face: mesh"
        elif face_mode == "bbox_only_possible_side_face":
            ftext = "Face: side bbox"
        elif face_mode == "pose_head_side_visible":
            ftext = "Face: head visible (no mesh)"
        else:
            ftext = "Face: not visible"
        fcolor = self.COLORS["normal"] if face_mode == "frontal_or_mesh" else self.COLORS["warning"]
        Y += ROW_H
        cv2.putText(frame, ftext, (LEFT_X, Y), self.FONT, self.FONT_SCALE_SMALL, fcolor, self.FONT_THICKNESS)

        # 3. Region
        region_data = detections.get("region", {})
        rstatus = region_data.get("status", "disabled")
        roverlap = 0.0
        if isinstance(region_data.get("features"), dict):
            roverlap = region_data.get("features", {}).get("body_overlap_ratio", 0)
        if rstatus == "in_region":
            rtext, rcolor = f"Region: in ({roverlap:.2f})", self.COLORS["normal"]
        elif region_data.get("exit_pending"):
            rtext, rcolor = f"Region: exiting {region_data.get('exit_pending_s', 0):.1f}s", self.COLORS["warning"]
        elif rstatus == "out_of_region":
            rtext, rcolor = f"Region: OUT ({roverlap:.2f})", self.COLORS["danger"]
        elif rstatus == "uncertain":
            rtext, rcolor = "Region: uncertain", self.COLORS["warning"]
        else:
            rtext, rcolor = "Region: off", self.COLORS["text"]
        Y += ROW_H
        cv2.putText(frame, rtext, (LEFT_X, Y), self.FONT, self.FONT_SCALE_SMALL, rcolor, self.FONT_THICKNESS)

        # Body/Torso/Head bboxes — always show when available
        if isinstance(region_data.get("features"), dict):
            feats = region_data["features"]
            for key, clr in [("body_bbox", self.COLORS["pose"]), ("torso_bbox", self.COLORS["torso"]), ("head_bbox", self.COLORS["head"])]:
                b = feats.get(key)
                if b:
                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), clr, 2)

        # 4. Occlusion
        occlusion_data = detections.get("occlusion", {})
        oconf = occlusion_data.get("confidence", 0.0)
        ostatus = occlusion_data.get("status", "unavailable")
        if ostatus in ("available", "fallback"):
            olabel = "Occlusion" if ostatus == "available" else "Occlusion(fb)"
            otext = f"{olabel}: {oconf:.2f}"
            ocolor = self.COLORS["danger"] if oconf > 0.6 else self.COLORS["warning"] if oconf > 0.3 else self.COLORS["normal"]
        else:
            otext = "Occlusion: N/A"
            ocolor = self.COLORS["text"]
        Y += ROW_H
        cv2.putText(frame, otext, (LEFT_X, Y), self.FONT, self.FONT_SCALE_SMALL, ocolor, self.FONT_THICKNESS)
        if oconf > 0.3 and isinstance(occlusion_data.get("features"), dict) and "roi_bbox" in occlusion_data["features"]:
            x1, y1, x2, y2 = occlusion_data["features"]["roi_bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), ocolor, 1)

        # 5. Exposure
        exposure_data = detections.get("limb_exposure", {})
        eratio = exposure_data.get("ratio", 0.0)
        efeats = exposure_data.get("features", {}) if isinstance(exposure_data.get("features"), dict) else {}
        elevel = efeats.get("coverage_level", "normal")
        estatus = exposure_data.get("status", "unavailable")
        if estatus == "unavailable":
            etext, ecolor = "Exposure: N/A", self.COLORS["text"]
        elif elevel == "body_or_legs_exposed":
            etext, ecolor = f"Exposure: BODY {eratio:.2f}", self.COLORS["danger"]
        elif elevel == "limb_exposed":
            limbs = ",".join(efeats.get("exposed_limbs", []))
            etext, ecolor = f"Exposure: {limbs} {eratio:.2f}", self.COLORS["warning"]
        elif estatus in ("uncertain_topology", "nearby_external_hand"):
            etext, ecolor = f"Exposure: {eratio:.2f} ({estatus})", self.COLORS["warning"]
        else:
            etext, ecolor = f"Exposure: ok {eratio:.2f}", self.COLORS["normal"]
        Y += ROW_H
        cv2.putText(frame, etext, (LEFT_X, Y), self.FONT, self.FONT_SCALE_SMALL, ecolor, self.FONT_THICKNESS)

        # 6. Cry (fused)
        cry_data = detections.get("cry", {})
        cconf = cry_data.get("confidence", 0.0)
        cstatus = cry_data.get("status", "unavailable")
        if cstatus == "available":
            ctext = f"Cry: {cconf:.2f}"
        elif cstatus in ("recent_hold",):
            ctext = f"Cry: {cconf:.2f} (recent)"
        elif cstatus in ("suspected_no_mesh", "suspected_mouth_no_mesh"):
            ctext = f"Cry: {cconf:.2f} (no mesh)"
        else:
            ctext = "Cry: N/A"
            ccolor = self.COLORS["text"]
        if cstatus != "unavailable":
            ccolor = self.COLORS["danger"] if cconf > 0.7 else self.COLORS["warning"] if cconf > 0.5 else self.COLORS["normal"]
        Y += ROW_H
        cv2.putText(frame, ctext, (LEFT_X, Y), self.FONT, self.FONT_SCALE_SMALL, ccolor, self.FONT_THICKNESS)

        # ═══════════════════════════════════════
        # 右列：传感器诊断
        # ═══════════════════════════════════════
        ry = 28
        target_fps = 15.0

        # FPS
        fps = results.get("fps", 0.0)
        if fps >= target_fps * 0.9:
            fpscolor = self.COLORS["normal"]
        elif fps >= target_fps * 0.5:
            fpscolor = self.COLORS["warning"]
        else:
            fpscolor = self.COLORS["danger"]
        cv2.putText(frame, f"FPS: {fps:.1f}", (RIGHT_X, ry), self.FONT, self.FONT_SCALE_SMALL, fpscolor, self.FONT_THICKNESS)

        # Audio
        audio = detections.get("audio", {})
        if audio:
            ry += ROW_H
            if not audio.get("gateway_healthy"):
                atext, acolor = "Audio: OFFLINE", self.COLORS["danger"]
            else:
                ac = audio.get("cry_confidence", 0)
                aseq = audio.get("audio_seq", 0)
                acolor = self.COLORS["danger"] if ac > 0.6 else self.COLORS["warning"] if ac > 0.35 else self.COLORS["normal"]
                atext = f"Aud: v{audio.get('volume',0):.2f} c{ac:.2f} r{audio.get('rhythm_score',0):.2f} b{audio.get('burst_count',0)} s{aseq}"
            cv2.putText(frame, atext, (RIGHT_X, ry), self.FONT, self.FONT_SCALE_SMALL, acolor, self.FONT_THICKNESS)

        # Temp
        cpu_temp = audio.get("cpu_temp_c") if audio else None
        if cpu_temp is not None:
            ry += ROW_H
            if cpu_temp >= 75:
                tcolor = self.COLORS["danger"]
            elif cpu_temp >= 65:
                tcolor = self.COLORS["warning"]
            else:
                tcolor = self.COLORS["normal"]
            cv2.putText(frame, f"CPU: {cpu_temp}C", (RIGHT_X, ry), self.FONT, self.FONT_SCALE_SMALL, tcolor, self.FONT_THICKNESS)

        # Motion
        motion = detections.get("motion", {})
        if motion:
            ry += ROW_H
            ag = motion.get("agitation", 0)
            moro = motion.get("moro_detected", False)
            if moro:
                mcolor = self.COLORS["danger"]
            elif ag > 0.3:
                mcolor = self.COLORS["warning"]
            else:
                mcolor = self.COLORS["text"]
            mtext = f"Motion: H{motion.get('head_motion',0):.2f} L{motion.get('limb_motion',0):.2f} A{ag:.2f}"
            if moro:
                mtext += " MORO"
            cv2.putText(frame, mtext, (RIGHT_X, ry), self.FONT, self.FONT_SCALE_SMALL, mcolor, self.FONT_THICKNESS)

        # ═══════════════════════════════════════
        # 底部：活跃告警
        # ═══════════════════════════════════════
        alert_y = Y + ROW_H + 6

        face_absence = detections.get("face_absence", {})
        if face_absence.get("status") == "not_visible":
            fdur = face_absence.get("duration_s", 0)
            fthr = face_absence.get("threshold_s", 15)
            facolor = self.COLORS["danger"] if fdur >= fthr else self.COLORS["warning"]
            cv2.putText(frame, f"FACE HIDDEN: {fdur:.1f}s", (LEFT_X, alert_y), self.FONT, self.FONT_SCALE_SMALL, facolor, self.FONT_THICKNESS)
            alert_y += ROW_H

        prone = detections.get("prone", {})
        if prone.get("status") == "suspected":
            cv2.putText(frame, f"PRONE RISK: {prone.get('duration_s', 0):.1f}s", (LEFT_X, alert_y), self.FONT, self.FONT_SCALE_SMALL, self.COLORS["danger"], self.FONT_THICKNESS)
            alert_y += ROW_H

        distress = detections.get("distress", {})
        if distress.get("confidence", 0) >= 0.35:
            dcolor = self.COLORS["danger"] if distress.get("confidence", 0) >= 0.7 else self.COLORS["warning"]
            cv2.putText(frame, f"Distress: {distress.get('confidence', 0):.2f} (muted)", (LEFT_X, alert_y), self.FONT, self.FONT_SCALE_SMALL, dcolor, self.FONT_THICKNESS)

        return frame

    def draw_safe_region(self, frame: np.ndarray, region_detector: RegionDetector) -> np.ndarray:
        """绘制安全区域"""
        if not self.show_safe_region:
            return frame

        return region_detector.draw_region(frame, self.COLORS["region"])

    def draw_status_bar(self, frame: np.ndarray, status: Dict) -> np.ndarray:
        """绘制状态栏"""
        h, w = frame.shape[:2]

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
        if status.get("has_prone", False):
            status_texts.append(("PRONE", self.COLORS["danger"]))
        if status.get("has_face_absence", False):
            status_texts.append(("HEAD/FACE HIDDEN", self.COLORS["warning"]))

        if not status_texts:
            return frame

        y_offset = 90
        for text, color in status_texts:
            (tw, th), _ = cv2.getTextSize(text, self.FONT, self.FONT_SCALE_SMALL, self.FONT_THICKNESS)
            cv2.putText(frame, text, (w - tw - 15, y_offset), self.FONT,
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
                text = f"CRY Detected (conf: {event.get('confidence', 0):.2f})"
            elif event_type == "occlusion_detected":
                text = f"OCCLUSION Detected (conf: {event.get('confidence', 0):.2f})"
            elif event_type == "limb_exposure":
                text = f"KICKED Blanket (ratio: {event.get('ratio', 0):.2f})"
            elif event_type == "region_exit":
                text = f"LEFT Safe Region"
            elif event_type == "face_not_visible":
                text = f"HEAD/FACE Hidden ({event.get('duration_s', 0):.1f}s)"
            else:
                text = f"Event: {event_type}"

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
            "Shortcuts:",
            "q: Quit",
            "h: Toggle Help",
            "d: Toggle Boxes",
            "r: Toggle Region",
            "s: Toggle Stats",
            "c: Calibrate Region"
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
        result_age_s = float(results.get("result_age_s", 0.0))
        stale_threshold = float(results.get("result_stale_threshold_s", 2.0))
        detections_fresh = result_age_s <= stale_threshold
        if not detections_fresh:
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Detection stale: {result_age_s:.1f}s", (w - 260, h - 170), self.FONT,
                        self.FONT_SCALE_NORMAL, self.COLORS["warning"], self.FONT_THICKNESS)

        if detections_fresh and "faces" in detections:
            frame = self.draw_face_detections(frame, detections["faces"])
        if detections_fresh and "hands" in detections:
            frame = self.draw_hand_detections(frame, detections["hands"], detections.get("occlusion"))
        if detections_fresh and "pose" in detections and detections["pose"] and "pose_data" in results:
            frame = self.draw_pose_landmarks(frame, results["pose_data"])

        # 绘制检测结果文字。结果过期时不继续显示旧 Region/Pose 状态，避免误导。
        if detections_fresh:
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
