# 音频哭闹检测集成指南

## 已完成的升级

**多模态哭闹识别系统** = **音频频谱分析** + **面部表情识别** + **肢体动作分析**

---

## 准确率提升预期

| 检测方式 | 准确率 | 误报率 | 说明 |
|---------|--------|--------|------|
| 纯视觉 | ~75% | 较高 | 依赖面部可见，角度不好易漏 |
| 纯音频 | ~80% | 中等 | 环境噪音易误报 |
| **音频+视觉融合** | **~95%+** | **极低** | 双模态确认，大幅提升可靠性 |

---

## 集成的功能

### 1. 音频检测器 (`src/audio_detector.py`)

**哭声特征提取：**
- **音量 RMS** - 哭声通常 > 0.1 (安静环境 < 0.01)
- **基频范围** - 婴儿哭声 400-1000Hz
- **频谱质心** - 800-3500Hz (哭声高频丰富)
- **高频能量占比** - 25%-70% (哭声特有频谱分布)

**实时处理：**
- 16kHz 采样率
- 100ms 处理块
- 8帧滑动窗口平滑

### 2. 音频网关 (`src/audio_gateway.py`)

**独立进程音频采集**：
- 50Hz 电源陷波 + 200Hz 高通降噪
- 频谱质心 (250-1500Hz 婴儿哭声频段)
- RMS 音量阈值
- P0 节律检测：哭声爆发周期 (0.4-1.3秒)，连续爆发计数 ≥3 次确认真哭

### 3. 多模态融合算法

```
三种检测路径（supervision.py）：

场景1：人脸可见 + presence确认
  → 面部表情(嘴巴张开+节律+眉毛) + 音频 + 动作 融合
  → 融合置信度 ≥ cry_confidence_threshold(0.5) 触发告警

场景2：in_region 但 landmarks 不可用（侧脸/盖被子）
  → 音频为主 + 动作辅助（有动作加成 +0.15）
  → 音频为主置信度 ≥ cry_confidence_threshold(0.5) 触发告警
  → 时间防抖：持续 1.5 秒

场景3：不在安全区域内
  → 纯音频路径（audio_only_cry_threshold=0.3）
```

### 4. 配置项 (`config.yaml`)

```yaml
audio:
  enabled: true
  cry_detection_enabled: true
  device_id: null
  process_mode: true
  sample_rate: 16000

detection:
  cry_confidence_threshold: 0.5
  cry_detection_enabled: true
  cry_duration_threshold: 2.0
  cry_sensitivity: high
```

---

## 告警输出字段

启用音频后，`cry_detected` 告警将包含额外字段：

```json
{
  "event_type": "cry_detected",
  "confidence": 0.92,
  "audio_confidence": 0.85,
  "visual_confidence": 0.78,
  "fused": true,
  "detection_mode": "audio_motion_only",
  "level": "danger"
}
```

---

## 真实场景优化

### 场景1：宝宝背对摄像头（面部不可见）
- **旧行为**：完全无法检测哭闹
- **新行为**：音频独立检测，置信度≥0.5 仍可触发告警（场景2降级路径）

### 场景2：宝宝张嘴打哈欠（不是哭）
- **旧行为**：易误报为哭闹
- **新行为**：音频无哭声信号 → 抑制误报

### 场景3：环境噪音（开关门、说话）
- **旧行为**：无影响
- **新行为**：频谱不匹配哭声特征 → 被过滤

### 场景4：宝宝小声抽泣
- **旧行为**：面部表情弱 → 可能漏报
- **新行为**：音频捕捉到哭声特征 → 补充检测

---

## 告警类型

当前系统支持 7 种告警类型，可通过 `calibrate_region.py` 或 `config.yaml` 配置：

| 告警类型 | 说明 | 触发条件 |
|---------|------|---------|
| `cry_detected` | 哭闹 | 多模态融合或音频路径置信度 ≥ 阈值 |
| `occlusion_detected` | 口鼻遮挡 | 手部/物体覆盖口鼻 ROI |
| `limb_exposure` | 踢被子/肢体裸露 | 肢体关键点高可见比例 |
| `region_exit` | 离开安全区域 | 身体与安全区域重叠率低于阈值 |
| `region_enter` | 进入安全区域 | 身体重新进入安全区域 |
| `prone_detected` | 趴睡风险 | 面部网格长时间不可见 |
| `face_not_visible` | 面部不可见 | 人脸和姿态头框均检测不到 |

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/audio_detector.py` | 音频核心检测逻辑（FFT特征提取） |
| `src/audio_gateway.py` | 独立进程音频采集 + 多模态融合 |
| `src/supervision.py` | 集成音频检测 + 三条检测路径 |
| `config.yaml` | 音频配置 + 告警类型配置 |
