"""
🎙️ 智能音频网关 - 独立进程模式
支持: 哭声检测、关键词检测、语音识别（后续扩展）

设计原则:
1. 永远不阻塞主推理进程
2. 崩溃自动重启，不影响主系统
3. 低优先级运行，CPU资源让给视觉算法
"""
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Process
from dataclasses import dataclass
from typing import Optional, Dict
import logging
import os
import psutil

logger = logging.getLogger(__name__)

# ==================== 数据结构 ====================

@dataclass
class AudioFeatures:
    """音频特征输出（传给主进程的唯一数据结构）"""
    timestamp: float
    rms_volume: float
    cry_confidence: float        # 综合哭声置信度
    cry_pattern_match: float     # 哭声模式匹配度
    is_crying: bool
    processing_latency_ms: float # 处理延迟（监控用）


# ==================== 哭声检测器（轻量版） ====================

class LightweightCryDetector:
    """
    轻量级哭声检测器（树莓派优化）
    计算量 < 1ms/次，CPU占用 <1%
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * 0.5)  # 500ms 帧

        # 哭声声学特征阈值（优化：降低音量阈值，提高灵敏度）
        self.volume_threshold = 0.008  # 从0.05降低，提高对小声哭闹的检测
        self.pitch_min = 250    # Hz，扩大范围
        self.pitch_max = 1500   # Hz，扩大范围
        self.centroid_min = 500 # Hz，扩大范围
        self.centroid_max = 4000 # Hz，扩大范围

        # 平滑窗口
        self.confidence_window = []
        self.window_size = 3

    def _compute_rms(self, audio: np.ndarray) -> float:
        """计算RMS音量"""
        return np.sqrt(np.mean(audio ** 2))

    def _compute_spectral_features(self, audio: np.ndarray) -> Dict:
        """计算频谱特征（快速FFT）"""
        try:
            n = len(audio)
            fft_vals = np.fft.rfft(audio * np.hanning(n))
            fft_freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)
            fft_mag = np.abs(fft_vals)

            if np.sum(fft_mag) < 1e-6:
                return {'centroid': 0, 'pitch': 0, 'hf_ratio': 0}

            # 频谱质心
            centroid = np.sum(fft_freqs * fft_mag) / np.sum(fft_mag)

            # 基频（哭声频段峰值）
            cry_mask = (fft_freqs >= self.pitch_min) & (fft_freqs <= self.pitch_max)
            if np.any(cry_mask):
                peak_idx = np.argmax(fft_mag[cry_mask])
                pitch = fft_freqs[cry_mask][peak_idx]
            else:
                pitch = 0

            # 高频能量占比（>1kHz）
            hf_mask = fft_freqs >= 1000
            hf_energy = np.sum(fft_mag[hf_mask] ** 2)
            total_energy = np.sum(fft_mag ** 2)
            hf_ratio = hf_energy / total_energy if total_energy > 0 else 0

            return {
                'centroid': centroid,
                'pitch': pitch,
                'hf_ratio': hf_ratio
            }
        except Exception as e:
            logger.debug(f"Spectral error: {e}")
            return {'centroid': 0, 'pitch': 0, 'hf_ratio': 0}

    def process_frame(self, audio: np.ndarray) -> AudioFeatures:
        """处理一帧音频，返回哭声特征"""
        start_time = time.time()

        volume = self._compute_rms(audio)
        spectral = self._compute_spectral_features(audio)

        # ========== 多特征加权置信度 ==========
        confidence = 0.0

        # 1. 音量权重 (35%)
        if volume > self.volume_threshold:
            volume_score = min(1.0, volume / 0.25)
            confidence += volume_score * 0.35

        # 2. 基频权重 (25%)
        if self.pitch_min <= spectral['pitch'] <= self.pitch_max:
            pitch_score = 1.0 - abs(spectral['pitch'] - 650) / 600
            confidence += max(0, pitch_score) * 0.25

        # 3. 频谱质心权重 (25%)
        if self.centroid_min <= spectral['centroid'] <= self.centroid_max:
            centroid_score = 1.0 - abs(spectral['centroid'] - 1800) / 1500
            confidence += max(0, centroid_score) * 0.25

        # 4. 高频占比权重 (15%)
        if 0.25 <= spectral['hf_ratio'] <= 0.7:
            hf_score = 1.0 - abs(spectral['hf_ratio'] - 0.45) / 0.35
            confidence += max(0, hf_score) * 0.15

        # 时间平滑
        self.confidence_window.append(confidence)
        if len(self.confidence_window) > self.window_size:
            self.confidence_window.pop(0)
        smoothed_confidence = sum(self.confidence_window) / len(self.confidence_window)

        # 模式匹配度（各特征同时满足的程度）
        pattern_match = 0.0
        conditions = [
            volume > self.volume_threshold,
            self.pitch_min <= spectral['pitch'] <= self.pitch_max,
            self.centroid_min <= spectral['centroid'] <= self.centroid_max,
            0.2 <= spectral['hf_ratio'] <= 0.75
        ]
        pattern_match = sum(conditions) / len(conditions)

        latency_ms = (time.time() - start_time) * 1000

        return AudioFeatures(
            timestamp=time.time(),
            rms_volume=float(volume),
            cry_confidence=float(smoothed_confidence),
            cry_pattern_match=float(pattern_match),
            is_crying=bool(smoothed_confidence >= 0.5),
            processing_latency_ms=float(latency_ms)
        )


# ==================== 音频网关主进程 ====================

class AudioGateway:
    """
    智能音频网关（独立进程模式）

    使用方式:
    >>> gateway = AudioGateway()
    >>> gateway.start()  # 启动独立进程，非阻塞!
    >>> features = gateway.get_latest_features()  # 获取最新特征，立即返回
    >>> gateway.stop()
    """

    def __init__(self, sample_rate: int = 16000, device_id: Optional[int] = None):
        self.sample_rate = sample_rate
        self.device_id = device_id
        self.enabled = False

        # 进程间通信（无锁队列）
        self._feature_queue: Queue = Queue(maxsize=5)  # 最多缓存5帧
        self._command_queue: Queue = Queue(maxsize=2)   # 控制命令
        self._process: Optional[Process] = None

        # 缓存的最新特征（队列为空时返回这个）
        self._latest_features = AudioFeatures(
            timestamp=0,
            rms_volume=0,
            cry_confidence=0,
            cry_pattern_match=0,
            is_crying=False,
            processing_latency_ms=0
        )

        # 健康监控
        self._last_update_time = 0
        self._health_status = "stopped"

        logger.info("AudioGateway initialized (process mode)")

    @staticmethod
    def _audio_process_loop(
        feature_queue: Queue,
        command_queue: Queue,
        sample_rate: int,
        device_id: Optional[int]
    ):
        """
        音频处理进程主循环（在独立进程中运行）
        永远不阻塞主系统！
        """
        # 设置进程为低优先级
        psutil.Process().nice(10)
        os.nice(10)

        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice not available, audio gateway disabled")
            return

        # 初始化检测器
        detector = LightweightCryDetector(sample_rate)

        # 环形音频缓冲区
        buffer_size = int(sample_rate * 0.5)  # 500ms
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        buffer_ptr = 0

        def audio_callback(indata, frames, time_info, status):
            """音频回调：仅拷贝数据，不做处理"""
            nonlocal audio_buffer, buffer_ptr
            if status:
                return

            data = indata.flatten() if indata.ndim > 1 else indata
            n = len(data)
            if buffer_ptr + n <= buffer_size:
                audio_buffer[buffer_ptr:buffer_ptr + n] = data
                buffer_ptr += n
            else:
                # 环形覆盖
                part1 = buffer_size - buffer_ptr
                audio_buffer[buffer_ptr:] = data[:part1]
                audio_buffer[:n - part1] = data[part1:]
                buffer_ptr = n - part1

        # 启动音频流
        try:
            stream = sd.InputStream(
                device=device_id,
                samplerate=sample_rate,
                channels=1,
                blocksize=int(sample_rate * 0.05),  # 50ms 小块
                callback=audio_callback,
                dtype='float32'
            )
            stream.start()
            logger.info(f"Audio stream started at {sample_rate}Hz")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            return

        # 主处理循环
        last_process_time = time.time()
        process_interval = 0.5  # 每500ms处理一次

        try:
            while True:
                # 检查退出命令
                if not command_queue.empty():
                    cmd = command_queue.get_nowait()
                    if cmd == "stop":
                        break

                now = time.time()
                # 定时处理音频
                if now - last_process_time >= process_interval:
                    # 处理音频，计算特征
                    features = detector.process_frame(audio_buffer.copy())

                    # 写入队列（队列满时丢弃旧数据）
                    if feature_queue.full():
                        try:
                            feature_queue.get_nowait()
                        except:
                            pass
                    try:
                        feature_queue.put_nowait(features)
                    except:
                        pass

                    last_process_time = now

                # 让出CPU
                time.sleep(0.05)

        finally:
            stream.stop()
            stream.close()
            logger.info("Audio process stopped gracefully")

    def start(self) -> bool:
        """启动音频网关（非阻塞，立即返回）"""
        if self._process and self._process.is_alive():
            logger.warning("AudioGateway already running")
            return True

        try:
            self._process = Process(
                target=self._audio_process_loop,
                args=(self._feature_queue, self._command_queue, self.sample_rate, self.device_id),
                daemon=True,  # 守护进程，主程序退出自动结束
                name="audio_gateway"
            )
            self._process.start()
            self._health_status = "running"
            self.enabled = True
            logger.info(f"AudioGateway process started (PID={self._process.pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to start AudioGateway: {e}")
            self._health_status = "error"
            return False

    def stop(self):
        """停止音频网关"""
        if self._process and self._process.is_alive():
            try:
                self._command_queue.put_nowait("stop")
                self._process.join(timeout=2)
                if self._process.is_alive():
                    self._process.terminate()
            except:
                pass
        self._health_status = "stopped"
        self.enabled = False
        logger.info("AudioGateway stopped")

    def get_latest_features(self) -> AudioFeatures:
        """
        获取最新音频特征（非阻塞，立即返回！）
        如果队列空，返回上次的特征
        永远不等待！
        """
        if not self.enabled:
            return self._latest_features

        # 取出所有新特征，只保留最新的
        latest = None
        while not self._feature_queue.empty():
            try:
                latest = self._feature_queue.get_nowait()
            except:
                break

        if latest is not None:
            self._latest_features = latest
            self._last_update_time = time.time()

        # 检查健康状态：超过3秒没更新 = 异常
        if time.time() - self._last_update_time > 3:
            self._health_status = "stale"

        return self._latest_features

    def is_healthy(self) -> bool:
        """检查音频网关是否健康运行"""
        if not self.enabled or not self._process:
            return False
        if not self._process.is_alive():
            return False
        if time.time() - self._last_update_time > 5:
            return False
        return True

    def get_health_status(self) -> Dict:
        """获取完整健康状态"""
        return {
            'enabled': self.enabled,
            'status': self._health_status,
            'process_alive': self._process.is_alive() if self._process else False,
            'last_update_ago_s': time.time() - self._last_update_time,
            'latest_confidence': self._latest_features.cry_confidence,
            'latest_volume': self._latest_features.rms_volume
        }


# ==================== 多模态融合（主进程侧） ====================

def multimodal_fusion(
    audio_features: AudioFeatures,
    visual_confidence: float,
    motion_confidence: float = 0.0,
    mouth_open: float = 0.0
) -> Dict:
    """
    多模态融合：音频 + 视觉 + 动作
    永远不阻塞！

    基于证据理论的简化实现：
    - 各证据源独立加权
    - 交叉确认加成
    - 单模态弱证据抑制
    """
    audio_conf = audio_features.cry_confidence
    audio_weight = 0.50
    visual_weight = 0.35
    motion_weight = 0.15

    # 加权基础融合
    fused = audio_conf * audio_weight + visual_confidence * visual_weight + motion_confidence * motion_weight

    # 交叉确认加成
    both_strong = audio_conf >= 0.5 and visual_confidence >= 0.5
    triple_strong = both_strong and (mouth_open >= 0.5 or motion_confidence >= 0.5)

    if triple_strong:
        fused = min(1.0, fused * 1.4)  # 三重确认 +40%
    elif both_strong:
        fused = min(1.0, fused * 1.25) # 双重确认 +25%
    elif audio_conf >= 0.7 and (mouth_open >= 0.5 or motion_confidence >= 0.3):
        fused = max(fused, audio_conf * 0.95)

    # 冲突抑制：音频强但视觉完全否定
    if audio_conf >= 0.6 and visual_confidence < 0.1:
        fused = fused * 0.6  # 降级处理

    # 确定置信等级
    if fused >= 0.7:
        level = "high"
    elif fused >= 0.4:
        level = "medium"
    elif fused >= 0.2:
        level = "low"
    else:
        level = "very_low"

    return {
        'fused_confidence': float(fused),
        'audio_confidence': float(audio_conf),
        'visual_confidence': float(visual_confidence),
        'motion_confidence': float(motion_confidence),
        'mouth_open': float(mouth_open),
        'certainty_level': level,
        'both_strong': both_strong,
        'triple_strong': triple_strong,
        'audio_volume': audio_features.rms_volume,
        'cry_pattern_match': audio_features.cry_pattern_match
    }
