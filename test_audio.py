#!/usr/bin/env python3
"""
音频哭闹检测器测试脚本
运行10秒，实时显示音频特征和哭声置信度
"""
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接导入，避免依赖mediapipe
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from audio_detector import AudioCryDetector


def main():
    print("=" * 60)
    print("🎤 婴儿哭闹音频检测器测试")
    print("=" * 60)

    detector = AudioCryDetector(sample_rate=48000, device_id=None)

    if not detector.enabled:
        print("❌ 音频模块不可用，请检查sounddevice安装")
        return 1

    if not detector.start_stream():
        print("❌ 启动音频流失败")
        return 1

    print("✅ 音频流启动成功，开始检测...")
    print("📊 实时音频特征 (按 Ctrl+C 停止)")
    print("-" * 60)
    print(f"{'时间':<8} {'音量':<10} {'基频':<8} {'质心':<8} {'高频占比':<10} {'哭声置信度':<12}")
    print("-" * 60)

    try:
        start_time = time.time()
        while time.time() - start_time < 10:  # 运行10秒
            confidence, details = detector.detect_cry()
            elapsed = int(time.time() - start_time)

            features = details.get("features", {})
            volume = features.get("volume", 0)
            pitch = features.get("pitch_frequency", 0)
            centroid = features.get("spectral_centroid", 0)
            hf_ratio = features.get("high_freq_ratio", 0)

            status = ""
            if confidence >= 0.7:
                status = "🔴 检测到哭声!"
            elif confidence >= 0.4:
                status = "🟡 疑似哭声"
            elif confidence >= 0.2:
                status = "🟢 有声音"

            print(f"{elapsed:<8} {volume:<10.3f} {pitch:<8.0f} {centroid:<8.0f} {hf_ratio:<10.2f} {confidence:<12.2%} {status}")
            time.sleep(0.3)

        print("-" * 60)
        print("\n✅ 测试完成!")
        print(f"   最后哭声置信度: {detector.last_cry_confidence:.2%}")
        print(f"   处理音频帧数: {detector.frame_count}")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        detector.stop_stream()

    return 0


if __name__ == "__main__":
    sys.exit(main())
