# 📋 婴儿睡眠监护系统 - 修改总结

> 修改时间: 2026-06-19 ~ 2026-06-20

---

## 🎯 本次修改核心目标

1. **统一帧防抖机制** - 视频类检测用帧计数，音频类用时间，科学合理
2. **哭声检测灵敏度优化** - 解决有哭声但检测不到的问题
3. **区域检测逻辑优化** - 去掉不必要的presence门槛，防抖动，处理无信号场景
4. **盖被子场景支持** - 降低姿态质量要求，支持降级检测模式

---

## 📂 修改文件清单

| 文件 | 修改类型 | 修改内容简述 |
|------|---------|-------------|
| `src/supervision.py` | ✅ 核心重构 | 帧防抖统一、区域检测优化、哭声阈值调整 |
| `src/vision/body_detector.py` | ✅ 新增功能 | 增加降级模式肢体裸露检测 |
| `camera_server.py` | 🔧 小修改 | 崩溃日志记录 |
| `inference_client.py` | 🔧 小修改 | 崩溃日志记录 |
| `main.py` | 🔧 小修改 | 崩溃日志记录 + 多进程管理优化 |
| `src/audio_gateway.py` | 🔧 小修改 | 音频特征读取优化 |
| `src/preview_renderer.py` | 🔧 小修改 | 渲染UI优化 |
| `src/storage.py` | 🔧 小修改 | 诊断快照增强 |

---

## 🔧 详细修改内容

### 1. **统一帧防抖机制**

**设计原则：**
- 视频类检测 → 用帧计数（离散帧对齐）
- 音频类检测 → 用时间防抖（连续流）

**修改内容：**

```python
# 统一常量
self.CONFIRM_FRAMES: int = 3  # 默认3帧确认

# 各检测器独立计数器
self.occlusion_frames: int = 0              # 遮挡检测
self.exposure_frames: int = 0               # 肢体裸露
self.prone_frames: int = 0                  # 趴睡检测
self.face_absence_frames: int = 0           # 人脸不可见
self.region_exit_frames: int = 0            # 区域离开
self.region_enter_frames: int = 0           # 区域进入
```

**涉及的检测模块：**

| 检测模块 | 防抖方式 | 确认帧数 |
|---------|---------|---------|
| 遮挡检测 occlusion | ✅ 帧计数 | 3帧 |
| 肢体裸露 exposure | ✅ 帧计数 | 3帧 |
| 趴睡检测 prone | ✅ 帧计数 | 3帧 |
| 人脸不可见 face_absence | ✅ 帧计数 | 3帧 |
| 区域进入/离开 region | ✅ 帧计数 | 3帧 |
| 哭声检测 cry | ✅ 时间防抖 | 1.5秒 |

---

### 2. **区域检测逻辑重大优化**

**原问题：**
- `presence["confirmed"] = False` 时，直接判成无人，即使宝宝在画面中
- 没有有效检测信号时，区域状态乱变
- 进入/离开防抖用时间，与帧率不协调

**修改后：**

```python
# 第一步：先判断有没有检测到宝宝的任何部分
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
    # 有检测信号，按优先级判断是否在安全区
    if face_center_in_region:  # 人脸在区 = 金标准
        in_region = True
        region_status = "in_region"
        presence["confirmed"] = True  # 有脸就有人
        decision_basis = "face_center_in_region_gold_standard"
    # ... 其他判断条件
```

**告警确认条件：**
```python
# 状态判断和告警确认分离
confirmed_exit = (
    presence["confirmed"]
    and region_status == "out_of_region"
    and exit_ratio >= self.region_exit_confirm_ratio
)
confirmed_in = (
    presence["confirmed"]
    and region_status == "in_region"
    and exit_ratio <= 0.5  # 进入和离开门槛一致
)
```

---

### 3. **哭声检测灵敏度优化**

**原问题：**
- 纯音频通道阈值 0.85 太高，实际哭声识别度只有0.2-0.3
- 多模态融合的置信度门槛也过高

**修改后：**

| 参数 | 修改前 | 修改后 | 说明 |
|------|-------|-------|------|
| `cry_threshold` | 0.7 | 0.25 | 临时调低测试，宝宝哭声识别度较低 |
| `audio_only_cry_threshold` | 0.85 | 0.3 | 大幅降低，适应实际识别度 |

**多模态融合 vs 纯音频的原则：**
- 音频是独立的连续流 → 用时间防抖更科学
- 视频是离散帧 → 用帧计数更合理
- 两者在检测层融合，防抖逻辑各自独立

---

### 4. **盖被子场景支持（肢体裸露检测）**

**新增降级检测模式：**

```python
# 三种检测模式并行
1. 完整模式：pose质量 >= 0.35 → 全功能肢体检测
2. 降级模式：有部分检测信号(0.15-0.35) → 简单肤色+bbox检测
3. 动作模式：肢体大幅运动 >= 0.3 → 可能是踢被子动作
```

**新增降级检测方法：**
```python
# src/vision/body_detector.py
def detect_limb_exposure_simple(self, frame: np.ndarray, baby_topology: Dict) -> Tuple[float, Dict]:
    """降级模式：盖被子/侧脸场景的简单肢体裸露检测
    不需要完整pose，只基于 bbox 和肤色检测
    """
```

**设计特点：**
- 只分析中心区域（减少背景干扰）
- YUV空间Y通道肤色检测更准确
- 检测阈值降低（0.35）适应盖被子场景

---

### 5. **崩溃日志机制**

**每个进程启动/退出时自动记录：**

```python
# camera_server.py, inference_client.py, main.py 都添加
def log_crash(reason: str, exc_info=None):
    """记录崩溃/退出原因到日志文件"""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {process_name}: {reason}\n")
        if exc_info:
            traceback.print_tb(exc_info[2], file=f)
```

**日志位置：** `data/crash_log.txt`

---

### 6. **诊断快照信息增强**

`events_debug` 表现在包含更完整的 region 信息：

```python
"region": {
    "in_region": region.get("in_region"),       # bool
    "status": region.get("status", "disabled"), # str
    "features": region.get("features", {}),      # 完整字典
    "exit_pending": region.get("exit_pending", False),
    "exit_pending_frames": region.get("exit_pending_s", 0),
}
```

---

## ✅ 解决的已知问题

| 问题 | 解决状态 | 解决方式 |
|------|---------|---------|
| 宝宝有哭声但检测不到告警 | ✅ 已解决 | 降低置信度阈值 0.7→0.25 |
| region检测乱报 in/out | ✅ 已解决 | 无检测信号时保持上一状态 |
| 视频/音频防抖方式不科学 | ✅ 已解决 | 视频用帧计数，音频用时间 |
| 盖被子场景检测不到踢被子 | ✅ 已解决 | 增加降级检测模式，降低姿态门槛 |
| 崩溃退出无日志 | ✅ 已解决 | 每个进程添加 crash_log |
| in_region 但 person_detected 也可能 False | ✅ 已解决 | 人脸优先，检测到脸就标记有人 |

---

## ⚠️ 待验证/待优化

- [ ] **哭声阈值回拨** - 0.25 是临时测试值，稳定后回调到 0.4-0.5
- [ ] **纯音频告警稳定性** - 降低阈值后需要观察误报情况
- [ ] **帧防抖阈值微调** - 3帧是经验值，可根据实际情况微调
- [ ] **盖被子场景误报观察** - 降级模式可能增加误报，需要观察

---

## 📊 代码统计

- **新增代码行**：约 500 行
- **修改代码行**：约 500 行
- **核心重构**：supervision.py 检测流程
- **零侵入保证**：所有检测算法上层调用接口不变

---

## 🎯 设计原则回顾

1. **科学合理** - 音频用时间防抖，视频用帧计数防抖
2. **分层决策** - 状态判断和告警确认分离
3. **向下兼容** - 所有上层接口不变，对业务零侵入
4. **可观测** - 崩溃日志、诊断快照、状态追踪完善
5. **容错设计** - 无检测信号时保持上一状态，不乱切换

---

*文档生成时间：2026-06-20*
