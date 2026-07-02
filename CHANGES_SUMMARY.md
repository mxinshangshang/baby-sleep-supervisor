# 📋 婴儿睡眠监护系统 - 修改总结

> 修改时间: 2026-06-19 ~ 2026-07-02

---

## 🎯 最近修改（2026-06-29）

### 1. **夜间哭声检测漏报修复** ✅

**问题根因：**
- 夜间模式哭声检测要求 `in_region AND presence["confirmed"]`
- 宝宝盖被子或动来动去时，`presence["confirmed"]` 可能通不过
- 纯音频快速通道只在 `not in_region` 时激活，造成检测间隙
- 23:00-23:05 持续有哭声但无任何告警

**修复方案：**
```python
# 夜间模式：不要求 presence/in_region
if not self.is_night_mode and not (presence["confirmed"] or in_region):
    # 白天模式：仍需要 presence 或 in_region 防误报
    ...
else:
    # 夜间模式：音频为主，动作为辅
    ...
```

**修改内容：**
| 文件 | 修改内容 |
|------|---------|
| `src/supervision.py` | 夜间模式哭声检测不再要求 presence/in_region |
| `src/audio_gateway.py` | 降低音量/频率阈值，更容易触发 is_crying |
| `config.yaml` | 调整安全区域坐标 |

**阈值调整：**
| 参数 | 修改前 | 修改后 |
|------|-------|-------|
| 夜间音频阈值 | 0.35 | 0.3 |
| volume_threshold | 0.008 | 0.005 |
| pitch_min/max | 250-1500Hz | 200-2000Hz |
| centroid_min/max | 500-4000Hz | 300-5000Hz |

---

### 2. **事件ID跳号严重问题修复** ✅

**根因：**
- `dual_camera_proxy.py` 调用 `save_event()` 时参数名错写为 `payload=`
- 应该是 `details=`
- SQLite AUTOINCREMENT 在 INSERT 尝试时就预分配 ID
- 即使 INSERT 失败，ID 也不会回收，导致 6901→6979→7157 大跨度跳号

**修复：**
```python
# src/camera/dual_camera_proxy.py
# 把两处 payload= 改为 details=
self._storage.save_event(..., details={...}, ...)
```

---

### 3. **标定程序优化** ✅

**改进内容：**
- 两列布局，避免字符覆盖
- 告警选择与检测开关**自动同步**
- 勾选什么告警就自动开启对应检测，没勾选就关闭
- 更友好的操作提示

---

## 📂 完整修改文件清单

| 文件 | 修改类型 | 修改内容简述 |
|------|---------|-------------|
| `src/supervision.py` | ✅ 核心修复 | 夜间哭声检测移除 presence/in_region 限制 |
| `src/audio_gateway.py` | ✅ 阈值优化 | 降低音频检测阈值 |
| `src/camera/dual_camera_proxy.py` | ✅ Bug修复 | payload→details 修复事件ID跳号 |
| `calibrate_region.py` | ✅ 功能优化 | 两列布局+告警检测自动同步 |
| `config.yaml` | 🔧 配置调整 | 安全区域坐标、禁用prone/limb_exposure |

---

## ✅ 已解决的问题汇总

| 问题 | 解决状态 | 解决方式 |
|------|---------|---------|
| 23:00-23:05 有哭声无告警 | ✅ 已解决 | 夜间模式不要求 presence/in_region |
| 事件ID跳号严重 (6901→7157) | ✅ 已解决 | 修复 dual_camera_proxy.py 参数名错误 |
| 标定程序界面布局问题 | ✅ 已解决 | 两列布局+自动同步检测开关 |
| 宝宝盖被子检测不到 | ✅ 已解决 | 降级检测模式+降低音频阈值 |
| 区域检测漏报夜间 | ✅ 已解决 | 之前已修复 elif→if |

---

## 🎯 设计原则回顾

1. **昼夜有别** - 夜间依赖音频，白天依赖视觉+音频
2. **容错设计** - 无检测信号时保持上一状态
3. **可观测性** - 诊断快照、状态追踪完善
4. **零侵入** - 检测算法上层调用接口不变

---

*文档更新时间：2026-07-02*
