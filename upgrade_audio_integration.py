#!/usr/bin/env python3
"""
音频集成升级脚本
自动修改 supervision.py 添加多模态哭闹检测支持
"""
import re
import shutil
from datetime import datetime


def backup_file(filepath: str) -> str:
    """创建备份"""
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ 已备份: {backup_path}")
    return backup_path


def apply_patch():
    """应用补丁"""
    filepath = "/home/mxin/.openclaw/workspace/baby_sleep_supervisor/src/supervision.py"

    # 1. 备份
    backup_file(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. 添加导入
    if 'from src.audio_detector' not in content:
        import_section = """from src.config import get_config
from src.vision.face_detector import FaceDetector"""

        new_import = """from src.config import get_config
from src.audio_detector import AudioCryDetector, fuse_audio_visual_cry
from src.vision.face_detector import FaceDetector"""

        content = content.replace(import_section, new_import)
        print("✅ 已添加音频模块导入")

    # 3. 在 __init__ 中添加音频检测器初始化
    init_section = """        # 告警冷却
        self.alert_cooldown = supervision_cfg.get("alert_cooldown_s", 60)
        self.last_alert_time: Dict[str, float] = {}"""

    new_init = """        # 告警冷却
        self.alert_cooldown = supervision_cfg.get("alert_cooldown_s", 60)
        self.last_alert_time: Dict[str, float] = {}

        # 音频哭闹检测器
        audio_cfg = self.config.get("audio", {})
        self.audio_enabled = audio_cfg.get("cry_detection_enabled", False)
        self.audio_detector = AudioCryDetector(
            sample_rate=audio_cfg.get("sample_rate", 48000),
            device_id=audio_cfg.get("device_id", None)
        )
        if self.audio_enabled:
            self.audio_detector.start_stream()
        self.last_audio_cry_confidence = 0.0
        self.audio_cry_window = deque(maxlen=8)
        self.audio_visual_fusion_enabled = True"""

    content = content.replace(init_section, new_init)
    print("✅ 已添加音频检测器初始化")

    # 4. 在 process_frame 中添加音频检测（在视觉cry检测前）
    cry_detection_section = """        if self.cry_enabled:
            if landmarks is not None:"""

    new_cry = """        # 音频哭闹检测
        audio_cry_confidence = 0.0
        audio_features = None
        if self.audio_enabled and self.audio_detector.is_running:
            audio_cry_confidence, audio_features = self.audio_detector.detect_cry()
            self.last_audio_cry_confidence = audio_cry_confidence
            results["detections"]["audio_cry"] = {
                "confidence": audio_cry_confidence,
                "features": audio_features
            }

        # 视觉哭闹检测
        if self.cry_enabled:
            if landmarks is not None:"""

    content = content.replace(cry_detection_section, new_cry)
    print("✅ 已添加音频检测调用")

    # 5. 在视觉cry检测完成后，添加多模态融合
    fusion_insert_point = """                if smoothed_cry >= 0.55:
                    self.last_cry_confidence = smoothed_cry
                    self.last_cry_time = now"""

    new_fusion = """                if smoothed_cry >= 0.55:
                    self.last_cry_confidence = smoothed_cry
                    self.last_cry_time = now

                # 多模态融合：音频 + 视觉
                if self.audio_enabled and self.audio_visual_fusion_enabled:
                    fused_cry, fusion_info = fuse_audio_visual_cry(
                        audio_cry_confidence,
                        smoothed_cry,
                        audio_features,
                        cry_features
                    )
                    cry_features.update(fusion_info)
                    smoothed_cry = fused_cry

                    results["detections"]["cry"].update({
                        "confidence": smoothed_cry,
                        "audio_confidence": audio_cry_confidence,
                        "visual_confidence": fusion_info["visual_confidence"],
                        "fused": True,
                        "fusion_reason": fusion_info["fusion_reason"]
                    })"""

    content = content.replace(fusion_insert_point, new_fusion)
    print("✅ 已添加多模态融合逻辑")

    # 6. 添加 stop 方法清理资源
    stop_section_pattern = r'def _bbox_from_points\(self'
    if 'def stop(self):' not in content:
        stop_method = """    def stop(self):
        \"\"\"停止监控，清理资源\"\"\"
        if self.audio_detector:
            self.audio_detector.stop_stream()

    def _bbox_from_points(self"""

        content = re.sub(r'def _bbox_from_points\(self', stop_method, content)
        print("✅ 已添加 stop 方法")

    # 7. 写入
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n🎉 所有补丁应用成功！")
    print("\n📝 升级完成后，请在 config.yaml 中启用音频：")
    print("   audio:")
    print("     enabled: true")
    print("     cry_detection_enabled: true")


if __name__ == "__main__":
    print("=" * 60)
    print("🔧 Baby Sleep Supervisor 音频集成升级")
    print("=" * 60)
    apply_patch()
