"""
音频哭闹检测器
基于音频特征的婴儿哭闹识别，与视觉识别融合提升准确率
"""
import time
import numpy as np
import threading
from collections import deque
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice not available, audio detection disabled")


class AudioCryDetector:
    """婴儿哭闹音频检测器"""

    def __init__(self, sample_rate: int = 48000, device_id: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device_id = device_id  # None = use default
        self.enabled = SOUNDDEVICE_AVAILABLE

        # 哭闹检测阈值
        self.cry_volume_threshold = 0.08  # 音量阈值
        self.cry_pitch_min = 300  # 哭声基频最小值 Hz
        self.cry_pitch_max = 1200  # 哭声基频最大值 Hz
        self.cry_spectral_centroid_min = 800
        self.cry_spectral_centroid_max = 3500

        # 时间平滑窗口
        self.volume_window = deque(maxlen=15)  # ~1.5秒
        self.pitch_window = deque(maxlen=10)
        self.cry_confidence_window = deque(maxlen=20)  # 从8扩大到20，支持节律分析
        self.confidence_timestamps = deque(maxlen=20)  # 记录每个置信度的时间
        self.audio_cry_start_time: Optional[float] = None
        self.last_cry_confidence = 0.0

        # 录音流
        self.audio_stream = None
        self.audio_buffer = deque(maxlen=100)  # 约10秒音频缓存
        self.lock = threading.Lock()

        # 运行状态
        self.is_running = False
        self.monitor_thread = None

        # 统计
        self.frame_count = 0
        self.last_process_time = time.time()

    def _audio_callback(self, indata, frames, time_info, status):
        """音频回调"""
        if status:
            logger.warning(f"Audio status: {status}")

        with self.lock:
            # 转换为单声道
            if indata.ndim > 1:
                audio_mono = np.mean(indata, axis=1)
            else:
                audio_mono = indata.flatten()

            self.audio_buffer.append(audio_mono.copy())

    def start_stream(self):
        """启动音频流"""
        if not self.enabled:
            logger.warning("Audio detection not available")
            return False

        try:
            self.audio_stream = sd.InputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=1,
                blocksize=int(self.sample_rate * 0.1),  # 100ms块
                callback=self._audio_callback
            )
            self.audio_stream.start()
            self.is_running = True
            logger.info(f"Audio stream started on device {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.enabled = False
            return False

    def stop_stream(self):
        """停止音频流"""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        self.is_running = False

    def _compute_rms_volume(self, audio: np.ndarray) -> float:
        """计算RMS音量"""
        return np.sqrt(np.mean(audio ** 2))

    def _compute_spectral_features(self, audio: np.ndarray) -> Dict:
        """计算频谱特征"""
        try:
            # FFT
            n = len(audio)
            fft_vals = np.fft.rfft(audio * np.hanning(n))
            fft_freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
            fft_mag = np.abs(fft_vals)

            # 频谱质心
            if np.sum(fft_mag) > 0:
                spectral_centroid = np.sum(fft_freqs * fft_mag) / np.sum(fft_mag)
            else:
                spectral_centroid = 0

            # 基频估计（简单峰值检测）
            pitch_freq = 0
            pitch_mask = (fft_freqs >= self.cry_pitch_min) & (fft_freqs <= self.cry_pitch_max)
            if np.any(pitch_mask):
                peak_idx = np.argmax(fft_mag[pitch_mask])
                pitch_freq = fft_freqs[pitch_mask][peak_idx]

            # 频谱带宽
            spectral_bandwidth = np.sqrt(np.sum(((fft_freqs - spectral_centroid) ** 2) * fft_mag) / np.sum(fft_mag)) if np.sum(fft_mag) > 0 else 0

            # 哭声特有：高频能量占比 (48kHz采样率，关注1kHz以上高频)
            high_freq_mask = fft_freqs >= 1000
            high_freq_energy = np.sum(fft_mag[high_freq_mask] ** 2)
            total_energy = np.sum(fft_mag ** 2)
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0

            return {
                "spectral_centroid": spectral_centroid,
                "pitch_frequency": pitch_freq,
                "spectral_bandwidth": spectral_bandwidth,
                "high_freq_ratio": high_freq_ratio,
                "total_energy": total_energy
            }
        except Exception as e:
            logger.debug(f"Spectral feature error: {e}")
            return {
                "spectral_centroid": 0,
                "pitch_frequency": 0,
                "spectral_bandwidth": 0,
                "high_freq_ratio": 0,
                "total_energy": 0
            }

    def _detect_cry_pattern(self, features: Dict) -> Tuple[float, Dict]:
        """检测哭声模式"""
        volume = features["volume"]
        spectral_centroid = features["spectral_centroid"]
        pitch_freq = features["pitch_frequency"]
        high_freq_ratio = features["high_freq_ratio"]

        confidence = 0.0
        reasons = []

        # 1. 音量评分
        if volume > self.cry_volume_threshold:
            volume_score = min(1.0, volume / 0.25)
            confidence += volume_score * 0.35
            reasons.append(f"volume={volume:.3f}")

        # 2. 基频评分（婴儿哭声通常在400-1000Hz）
        if self.cry_pitch_min <= pitch_freq <= self.cry_pitch_max:
            pitch_score = 1.0 - abs(pitch_freq - 650) / 600  # 650Hz最优
            confidence += pitch_score * 0.25
            reasons.append(f"pitch={int(pitch_freq)}Hz")

        # 3. 频谱质心评分
        if self.cry_spectral_centroid_min <= spectral_centroid <= self.cry_spectral_centroid_max:
            centroid_score = 1.0 - abs(spectral_centroid - 1800) / 1500
            confidence += max(0, centroid_score) * 0.25
            reasons.append(f"centroid={int(spectral_centroid)}Hz")

        # 4. 高频能量占比（哭声有明显高频分量）
        if 0.25 <= high_freq_ratio <= 0.7:
            hf_score = 1.0 - abs(high_freq_ratio - 0.45) / 0.35
            confidence += max(0, hf_score) * 0.15
            reasons.append(f"hf_ratio={high_freq_ratio:.2f}")

        return confidence, {
            "reasons": reasons,
            "features": features
        }

    def detect_cry(self) -> Tuple[float, Dict]:
        """执行哭声检测"""
        if not self.enabled or not self.is_running:
            return 0.0, {"status": "audio_disabled"}

        with self.lock:
            if len(self.audio_buffer) < 5:  # 需要至少500ms数据
                return 0.0, {"status": "buffering", "buffer_size": len(self.audio_buffer)}

            # 合并最近500ms音频
            audio = np.concatenate(list(self.audio_buffer)[-5:])

        # 计算特征
        volume = self._compute_rms_volume(audio)
        spectral_features = self._compute_spectral_features(audio)

        features = {
            "volume": volume,
            **spectral_features
        }

        # 检测哭声
        raw_confidence, details = self._detect_cry_pattern(features)

        # 时间平滑
        smoothed_confidence = self._get_smoothed_confidence(raw_confidence)

        # ========== 节律特征增强 ==========
        # 对「一声间隔一声」的典型婴儿哭声做针对性增强
        rhythm_boost = self._compute_cry_rhythm_score()
        rhythm_enhanced = min(1.0, smoothed_confidence * rhythm_boost)

        # 只增强中间置信度区间：太低肯定不是，太高也没必要增强
        # 重点解决：节律性哭声被平均稀释后置信度不够的问题
        if 0.35 <= smoothed_confidence <= 0.8:
            final_confidence = rhythm_enhanced
        else:
            final_confidence = smoothed_confidence

        self.last_cry_confidence = final_confidence
        self.frame_count += 1

        return final_confidence, {
            "raw_confidence": raw_confidence,
            "smoothed_confidence": smoothed_confidence,
            "rhythm_boost": rhythm_boost,
            "final_confidence": final_confidence,
            "features": features,
            "details": details
        }

    def _get_smoothed_confidence(self, new_confidence: float) -> float:
        """获取平滑后的置信度"""
        self.cry_confidence_window.append(new_confidence)
        return sum(self.cry_confidence_window) / len(self.cry_confidence_window)

    def _compute_cry_rhythm_score(self) -> float:
        """
        计算哭声节律得分
        婴儿哭声特征：一声间隔一声，0.5-2秒周期，重复性强
        算法完全基于统计，O(1)开销，零延迟
        """
        window = list(self.cry_confidence_window)
        if len(window) < 8:  # 需要至少4秒数据
            return 1.0  # 数据不足时不做惩罚

        # ========== 1. 峰值检测 ==========
        peak_indices = []
        peak_values = []
        for i in range(1, len(window) - 1):
            # 局部峰值：比左右都高，且绝对幅度足够
            if window[i-1] < window[i] > window[i+1] and window[i] >= 0.4:
                peak_indices.append(i)
                peak_values.append(window[i])

        peak_count = len(peak_indices)
        if peak_count < 2:  # 峰值太少，无节律
            return 1.0

        # ========== 2. 峰值间隔规律性 ==========
        # 每500ms一个采样点，典型间隔1-4个点（0.5-2秒）
        peak_intervals = np.diff(peak_indices)
        mean_interval = np.mean(peak_intervals)
        std_interval = np.std(peak_intervals) if len(peak_intervals) > 1 else 999

        # 间隔评分：1-3个点（0.5-1.5秒）最优
        interval_score = 0.0
        if 1.0 <= mean_interval <= 4.0:
            interval_score = 1.0 - abs(mean_interval - 2.5) / 3.0
            interval_score = max(0.3, interval_score)

        # 间隔稳定性评分：标准差越小越规律
        stability_score = 1.0 - min(1.0, std_interval / 2.0)
        stability_score = max(0.4, stability_score)

        # ========== 3. 占空比（哭-停-哭-停模式）==========
        high_count = sum(1 for c in window if c >= 0.5)
        duty_cycle = high_count / len(window)
        # 典型哭声占空比：30-60%（有哭有停，不是连续也不是太少）
        duty_cycle_score = 0.0
        if 0.25 <= duty_cycle <= 0.75:
            duty_cycle_score = 1.0 - abs(duty_cycle - 0.5) / 0.5
            duty_cycle_score = max(0.5, duty_cycle_score)

        # ========== 4. 峰值幅度一致性 ==========
        peak_std = np.std(peak_values) if len(peak_values) > 1 else 0
        peak_consistency_score = 1.0 - min(1.0, peak_std / 0.3)
        peak_consistency_score = max(0.5, peak_consistency_score)

        # ========== 5. 峰值频率（单位时间内的峰值数量）==========
        # 10秒内有3-8个峰值是典型的哭声节律
        peak_frequency = peak_count / len(window) * 20  # 归一化到10秒
        frequency_score = 0.0
        if 2.0 <= peak_frequency <= 10.0:
            frequency_score = 1.0 - abs(peak_frequency - 5.0) / 5.0
            frequency_score = max(0.4, frequency_score)

        # ========== 融合节律增强因子 ==========
        rhythm_weight = 0.4 * interval_score + \
                        0.25 * stability_score + \
                        0.15 * duty_cycle_score + \
                        0.1 * peak_consistency_score + \
                        0.1 * frequency_score

        # 节律因子范围：0.7（无节律惩罚）~ 1.5（强节律大幅增强）
        rhythm_boost = 0.7 + rhythm_weight * 0.8

        return rhythm_boost

    def get_status(self) -> Dict:
        """获取状态"""
        return {
            "enabled": self.enabled,
            "running": self.is_running,
            "buffer_size": len(self.audio_buffer) if self.enabled else 0,
            "last_cry_confidence": self.last_cry_confidence,
            "frame_count": self.frame_count
        }


def fuse_audio_visual_cry(
    audio_confidence: float,
    visual_confidence: float,
    audio_features: Optional[Dict] = None,
    visual_features: Optional[Dict] = None,
    motion_confidence: float = 0.0,
    mouth_open_score: float = 0.0
) -> Tuple[float, Dict]:
    """
    基于证据理论(D-S)的多模态哭闹置信度融合

    科学依据：
    - 婴儿哭闹是"声学特征 + 面部表情 + 肢体动作
    三个独立证据源，具有互补性

    融合层次：
    1. 强证据(>70%) → 强信号，权重高
    2. 中证据(40-70%) → 需要交叉验证
    3. 弱证据(<40%) → 作为辅助，不单独触发
    """
    # ========== 证据权重分配 ==========
    # 音频证据 (最强，对应哭声本身
    audio_strong = audio_confidence >= 0.7
    audio_medium = 0.4 <= audio_confidence < 0.7
    audio_weak = audio_confidence < 0.4

    # 视觉证据 (表情)
    visual_strong = visual_confidence >= 0.7
    visual_medium = 0.4 <= visual_confidence < 0.7
    visual_weak = visual_confidence < 0.4

    # 动作证据 (辅助)
    has_motion = motion_confidence >= 0.3
    mouth_open = mouth_open_score >= 0.5

    # ========== 融合决策 ==========
    evidence_count = sum([
        audio_strong, visual_strong, audio_medium and (has_motion or mouth_open)])

    # 基础融合：加权平均
    audio_weight = 0.50
    visual_weight = 0.35
    motion_weight = 0.15

    fused = (audio_confidence * audio_weight +
             visual_confidence * visual_weight +
             motion_confidence * motion_weight)

    fusion_reason = []
    certainty_level = "low"

    # ========== 证据增强规则 ==========

    # 【最高置信度：双强证据 → 几乎100%确认
    if audio_strong and visual_strong:
        fused = min(1.0, fused * 1.4)
        fusion_reason.append("dual_strong_evidence")
        certainty_level = "very_high"

    # 【高置信度】：音频强 + (视觉中 或 张嘴/动作)
    elif audio_strong and (visual_medium or mouth_open or has_motion):
        fused = max(fused, audio_confidence * 0.95)
        fusion_reason.append("audio_strong_with_support")
        certainty_level = "high"

    # 【高置信度】：视觉强 + (音频中 或 动作)
    elif visual_strong and (audio_medium or has_motion):
        fused = max(fused, visual_confidence * 0.90)
        fusion_reason.append("visual_strong_with_support")
        certainty_level = "high"

    # 【中置信度】：双中证据 + 辅助证据
    elif audio_medium and visual_medium and (mouth_open or has_motion):
        fused = min(1.0, fused * 1.2)
        fusion_reason.append("dual_medium_with_support")
        certainty_level = "medium_high"

    # 【中置信度】：双中证据
    elif audio_medium and visual_medium:
        fused = min(1.0, fused * 1.1)
        fusion_reason.append("dual_medium_evidence")
        certainty_level = "medium"

    # ========== 证据冲突抑制 ==========

    # 【低置信度抑制】：单证据弱，另一证据也弱
    if audio_weak and visual_weak:
        fused = min(fused, 0.15)
        fusion_reason.append("both_weak_suppressed")
        certainty_level = "very_low"

    # 【单模态抑制】：只有单一弱证据，无其他支持
    elif audio_weak and not visual_strong and not has_motion:
        fused = min(fused, visual_confidence * 0.7)
        fusion_reason.append("audio_weak_suppressed")

    elif visual_weak and not audio_strong:
        fused = min(fused, audio_confidence * 0.7)
        fusion_reason.append("visual_weak_suppressed")

    # 【冲突处理】：音频强但视觉完全否定
    if audio_strong and visual_confidence < 0.1:
        fused = fused * 0.6  # 降级处理，可能是环境噪音
        fusion_reason.append("audio_visual_conflict_downgraded")

    return fused, {
        "audio_confidence": audio_confidence,
        "visual_confidence": visual_confidence,
        "motion_confidence": motion_confidence,
        "mouth_open": mouth_open,
        "fused_confidence": fused,
        "fusion_reason": fusion_reason,
        "certainty_level": certainty_level,
        "evidence_count": evidence_count,
        "audio_strong": audio_strong,
        "visual_strong": visual_strong
    }
