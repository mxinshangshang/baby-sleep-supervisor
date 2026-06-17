# 系统架构总览 - Baby Sleep Supervisor v1.0

---

## 整体架构

```
                          ┌─────────────────────────────────────────────────┐
                          │              main.py 主进程                     │
                          │   (双进程管理 + 自动重启 + 信号处理)            │
                          └─────────────┬───────────────────┬──────────────┘
                                        │                   │
                          ┌─────────────▼───────┐   ┌───────▼──────────────┐
                          │ camera_server.py    │   │ inference_client.py  │
                          │ 摄像头采集进程      │   │ 推理客户端进程       │
                          │ (系统Python)        │   │ (Python 3.11 venv)   │
                          └─────────┬───────────┘   └───────────┬──────────┘
                                    │                           │
                                    │  TCP:65433               │
                                    │  BGR888 零拷贝          │
                                    └───────────────────────────┘
```

---

## 核心进程解耦设计

### 1. 摄像头进程 (camera_server.py)
**运行环境**: 系统 Python 3.x (`/usr/bin/python3`)

**职责**:
- 独占摄像头硬件，使用 `picamera2` 原生接口采集
- 支持多种像素格式：BGR888, RGB888, YUV420
- 固定帧率推送，避免推理波动影响采集稳定性
- TCP Socket 服务端，帧头：4字节 little-endian 长度 + 原始像素数据

**关键优化**:
```python
# 全传感器视野模式 - Camera Module 3 Wide 专用
use_full_sensor_fov: true

# 默认 960x540 BGR888 15fps
# 与 OpenCV 直接兼容，零拷贝
```

---

### 2. 推理进程 (inference_client.py)
**运行环境**: Python 3.11 虚拟环境 (`kid_supervisor_v3`)

**核心设计模式：三流解耦**

```
┌─────────────────────────────────────────────────────────────────┐
│                     三线程并行解耦架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────┐ │
│  │  接收线程        │    │  算法线程        │    │  UI线程    │ │
│  │  (LatestFrame-  │    │  (LatestInferen- │    │  (Main)    │ │
│  │   Receiver)      │    │   ceWorker)      │    │            │ │
│  │                  │    │                  │    │            │ │
│  │  持续读Socket    │    │  inference_fps   │    │ display_fps│ │
│  │  只保留最新帧    │    │  可控帧率检测    │    │  画面渲染   │ │
│  └────────┬─────────┘    └────────┬─────────┘    └──────┬─────┘ │
│           │                       │                       │       │
│           ▼                       ▼                       ▼       │
│    latest_frame             detection results          render   │
│    (无锁快照)              (原子更新)                (无阻塞)    │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

**三流独立帧率**:
| 流类型 | 配置项 | 默认值 | 说明 |
|--------|--------|--------|------|
| 摄像头采集 | `camera.fps` | 15fps | 固定，不动态调整 |
| 算法推理 | `inference.inference_fps` | 3fps | 高温时自动降频 |
| UI预览 | `preview.display_fps` | 15fps | 始终流畅，不受算法影响 |

---

## 监督核心 (supervision.py)

**代码量**: 1513行，占总代码 31%

### 检测流水线

```
输入帧
  ↓
┌──────────────────────────────────────────────────────────┐
│  第一阶段：基础检测器并行                                 │
├──────────────────┬──────────────────┬──────────────────┤
│  FaceDetector    │  BodyDetector    │  AudioGateway    │
│  (人脸+手部)     │  (姿态骨架)      │  (音频哭声)      │
└──────────────────┴──────────────────┴──────────────────┘
  ↓
┌──────────────────────────────────────────────────────────┐
│  第二阶段：特征聚合                                       │
│  _summarize_face()  +  _summarize_pose()                │
│  → presence (婴儿存在确认)                               │
│  → baby_topology (身体拓扑)                             │
│  → motion_features (肢体运动分析)                        │
└──────────────────────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────────────────────┐
│  第三阶段：专项检测器                                     │
├─────────────┬──────────────┬──────────────┬─────────────┤
│  哭闹检测    │  遮挡检测    │  裸露检测    │  区域检测    │
│  +趴睡      │  +面部缺失   │              │             │
└─────────────┴──────────────┴──────────────┴─────────────┘
  ↓
┌──────────────────────────────────────────────────────────┐
│  第四阶段：多模态证据融合 + 防抖 + 告警决策                │
│  _combine_cry_evidence()                                │
│  multimodal_fusion()                                    │
│  _should_alert() 冷却/频率控制                           │
└──────────────────────────────────────────────────────────┘
  ↓
输出结果 + 通知
```

---

### 1. 婴儿存在确认 (Presence Detection)

**核心逻辑**: `_compute_presence()`

**输入**:
- 人脸检测结果（数量、置信度、面积）
- 姿态检测结果（关键点数量、躯干框面积、核心可见性）

**算法**:
```
滑窗 8 帧 → 存在信号比例 ≥ 75% → 确认为"婴儿在视野中"

不确定状态有 2秒 grace period，避免帧间抖动误触发"离开区域"。
```

**重要**: `alert_requires_confirmed_presence=true`
- 所有告警必须在确认婴儿存在后才会触发
- 防止空床、光线变化等导致的误报

---

### 2. 身体拓扑推断 (Baby Topology)

**核心逻辑**: `_compute_baby_topology()`

**目标**: 从 MediaPipe Pose 33个关键点推断婴儿身体朝向和姿态

```
关键点分组：
  头部：0-10 (脸+耳朵)  → head_bbox 橙框
  躯干：11,12,23,24    → torso_bbox 紫框

身体朝向判断：
  face_mode == "bbox_only_possible_side_face"
  + pose_head_bbox 与 face_bbox 一致性好
  + yaw_ratio 偏侧 → "侧卧"
```

---

### 3. 哭闹检测 (Cry Detection)

**四层证据链 + 多模态融合**:

```
第1层：面部表情 (face_detector.py)
  ├─ 嘴巴张开持续时间
  ├─ 嘴巴开合节律
  └─ 眼部/眉毛形态

第2层：肢体运动 (_compute_motion_features())
  ├─ 头部摆动幅度
  ├─ 手臂运动幅度
  ├─ 整体躁动程度
  └─ Moro 惊跳反射识别（避免误判）

第3层：音频声学 (audio_gateway.py)
  ├─ 50Hz 电源陷波 + 200Hz 高通降噪
  ├─ 频谱质心 (250-1500Hz 婴儿哭声频段)
  └─ RMS 音量阈值

第4层：音频节律 (P0节律检测)
  ├─ 哭声爆发周期 (0.4-1.3秒)
  ├─ 连续爆发计数 (≥3次确认真哭)
  └─ 周期间隔一致性评分

融合层：multimodal_fusion()
  声学分(50%) + 节律分(50%) → 最终置信度
  视觉置信度与音频置信度加权融合
```

---

### 4. 口鼻遮挡检测 (Occlusion Detection)

**核心逻辑**: `face_detector.detect_occlusion()`

```
步骤：
  1. Face Mesh 提取 4个关键点 (鼻+嘴三角区)
  2. 裁剪 ROI → 灰度化 → 局部二值模式
  3. 纹理模糊度 + 边缘密度 → 遮挡评分
  4. 手部 bbox 与口鼻 ROI 重叠 → 风险加成

降级路径：
  Face Mesh 可用 → 精准纹理分析
  Face Mesh 不可用 → fallback 肤色+边缘
  完全不可见 → 面部缺失检测接管
```

---

### 5. 趴睡/面部朝下检测 (Prone Detection)

**触发条件**:
```
face_mode == "pose_head_side_visible" (只有姿态头框，没有人脸网格)
持续 ≥ face_mesh_absence_duration_threshold (默认 8秒)
→ 判定为"面部朝下 / 趴睡风险"
```

---

### 6. 肢体裸露检测 (Limb Exposure)

**核心逻辑**: 基于身体拓扑 + 关键点可见性

```
原理：
  正常盖被子 → 腿/手臂关键点被遮挡，置信度低
  踢被子 → 皮肤裸露，MediaPipe 能检测到高置信度关键点

判断：
  下肢关键点(25,26,27,28)可见比例
  + 上肢关键点(11-16)可见比例
  → exposure_ratio 阈值 0.35
  → 持续 ≥ 8秒 告警

排除：外部人手靠近不算踢被子
```

---

### 7. 区域检测 (Region Detection)

**支持格式**:
- 旧版：矩形 2个对角点
- 新版：多边形 N个顶点（默认4点，婴儿床四边形）

**判断逻辑**:
```
身体框 bbox 与安全区域重叠率：
  body_overlap < 0.55 + torso_overlap < 0.50
  持续 ≥ region_exit_confirm_ratio (33% 的 6帧窗口)
  → 判定离开区域

边缘触发：
  只有状态翻转 (in→out / out→in) 才发通知
  不是持续在区域内或外就一直通知

预缓存抓拍：
  状态刚变化时立即存帧
  防抖确认后用缓存帧通知，确保画面与事件瞬间同步
```

---

## 颜色编码系统 (Preview Renderer)

| 颜色 | 元素 | 来源 | 说明 |
|------|------|------|------|
| 🔵 蓝 | Face Box | MediaPipe Face Detection | 检测到的人脸 |
| 🟠 橙 | Head Box | MediaPipe Pose 0-10 | 姿态拟合的头部范围 |
| 🟣 紫 | Torso Box | MediaPipe Pose 11,12,23,24 | 姿态拟合的躯干范围 |
| 🟡 黄 | Hand Nearby | MediaPipe Hands | 手在头部附近 |
| 🔴 红 | Hand Occluder | MediaPipe Hands + 重叠判断 | 手遮挡口鼻，高风险 |
| 🟢 绿 | Safe Region | 手工标定 | 安全睡眠区域 |
| ⚪ 青 | Pose Skeleton | MediaPipe Pose | 姿态骨架点和连线 |

---

## 温控与性能调节

**模块**: `thermal` 配置 + inference_client.py 动态调整

```
温度阈值：
  ≥ 65°C → warning，开始降频到 throttle_inference_fps (默认2fps)
  ≥ 75°C → danger，模型复杂度从 1 降到 0 (Lite版)

调整原则：
  ✓ 只降低算法帧率，绝不降低预览帧率
  ✓ 只降低模型复杂度，不减少检测功能
  ✓ 温度恢复后自动复原
```

---

## 事件存储与生命周期

**模块**: storage.py + SQLite

```
事件表：
  id, timestamp, type, level, confidence, photo_path, details

自动清理：
  event_retention_days: 30 天
  max_photo_size_mb: 1024 MB
```

---

## 容错与自愈

| 机制 | 说明 |
|------|------|
| 进程自动重启 | max_restart_attempts=5次，退避2,4,6,8,10秒 |
| 进程隔离 | 摄像头崩溃不影响推理，反之亦然 |
| 检测结果过期 | detection_stale > 阈值 → 停止显示旧状态，避免误导 |
| 存在确认缓冲 | 所有告警前置 presence 确认，空床不报 |
| 结果防抖 | 所有判断都有时序窗口，不依赖单帧 |

---

## 配置参数总览

| 模块 | 关键参数 | 默认值 |
|------|----------|--------|
| 存在确认 | presence_window_size | 8帧 |
| | presence_confirm_ratio | 75% |
| 哭闹 | cry_confidence_threshold | 0.5 |
| | cry_duration_threshold | 2.0秒 |
| 遮挡 | occlusion_threshold | 0.6 |
| | occlusion_duration_threshold | 1.0秒 |
| 裸露 | exposure_threshold | 0.35 |
| | exposure_duration_threshold | 8.0秒 |
| 区域 | region_body_overlap_threshold | 0.55 |
| | region_exit_window_size | 6帧 |
| 趴睡 | face_mesh_absence_duration_threshold | 8.0秒 |
| 冷却 | alert_cooldown_s | 60秒 |

---

## 代码统计

| 文件 | 行数 | 占比 | 职责 |
|------|------|------|------|
| src/supervision.py | 1513 | 31% | 监督核心，所有检测逻辑 |
| src/audio_gateway.py | 581 | 12% | 音频网关 + 哭声检测 |
| src/vision/face_detector.py | 631 | 13% | 人脸+表情+手部+遮挡 |
| src/preview_renderer.py | 457 | 9% | UI渲染 + 颜色编码 |
| src/audio_detector.py | 449 | 9% | 音频信号处理 |
| src/notifier.py | 393 | 8% | 飞书通知 |
| src/storage.py | 387 | 8% | SQLite存储 |
| src/vision/body_detector.py | 204 | 4% | 姿态检测 |
| src/vision/region_detector.py | 158 | 3% | 区域判断 |
| src/config.py | 54 | 1% | 配置管理 |
| **总计** | **4846** | **100%** | |
