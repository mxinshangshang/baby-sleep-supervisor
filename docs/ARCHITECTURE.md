# 系统架构总览 - Baby Sleep Supervisor

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
                          │                     │   │                      │
                          │ ┌─────────────────┐ │   │                      │
                          │ │ CameraRouter    │ │   │                      │
                          │ │ (双摄互斥路由)  │ │   │                      │
                          │ └─────────────────┘ │   │                      │
                          └─────────┬───────────┘   └───────────┬──────────┘
                                    │                           │
                                    │  TCP:65433               │
                                    │  RGB888 零拷贝          │
                                    └───────────────────────────┘
```

---

## 核心进程解耦设计

### 1. 摄像头进程 (camera_server.py)
**运行环境**: 系统 Python 3.x (`/usr/bin/python3`)

**职责**:
- 独占摄像头硬件，使用 `picamera2` 原生接口采集
- 支持双摄像头自动切换（常规 imx708_wide ↔ 夜视 imx708_wide_noir）
- 固定帧率推送，避免推理波动影响采集稳定性
- TCP Socket 服务端，帧头：4字节 little-endian 长度 + 原始 BGR 像素数据
- 日志重定向到 `/tmp/camera_server.log`，`PYTHONUNBUFFERED=1` 确保实时输出

**双摄路由 (CameraRouter)**:
- 对外暴露与 Picamera2 同名的鸭子接口 (`capture_array()`, `camera_configuration()`)
- 亮度检测基于图像中心 50% 区域灰度均值归一化到 0-1
- 迟滞比较器防抖：`night_threshold=0.15`（低于此切夜视），`day_threshold=0.30`（高于此切回）
- `stable_frames=7`（15fps 下约 0.5s 确认），`min_switch_interval=300s`（防来回抖动）
- 切换时新旧摄像头 `warmup_overlap_s=2.0s` 重叠，异步关闭旧摄像头
- 线程安全，`_switch_lock` 防止并发切换

**关键优化**:
```python
# 全传感器视野模式 - Camera Module 3 Wide 专用
use_full_sensor_fov: true

# 默认 960x540 RGB888 15fps
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

## 模块懒加载 (src/__init__.py)

使用 PEP 562 `__getattr__` 实现按需导入，防止 `camera_server.py` 导入 `src` 包时触发 `mediapipe` 等重型依赖加载：

```python
_LAZY_EXPORTS = {
    'AudioCryDetector': '.audio_detector',
    'fuse_audio_visual_cry': '.audio_detector',
    'get_config': '.config',
    'load_config': '.config',
    'save_config': '.config',
    'Notifier': '.notifier',
    'Storage': '.storage',
    'SleepSupervisor': '.supervision',
}

def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module = __import__(_LAZY_EXPORTS[name], fromlist=[name], level=1)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(...)
```

**设计原因**: `camera_server.py`（系统 Python）仅需要 `CameraRouter`，不应加载 `mediapipe`。`CameraRouter` 通过 `from src.camera.dual_camera_proxy import CameraRouter` 直接导入子模块，绕过 `src/__init__.py`。

---

## 监督核心 (supervision.py)

**代码量**: 1852行

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

三种检测路径：
  场景1：人脸可见 + presence确认 → 面部表情 + 音频 + 动作融合
  场景2：in_region但landmarks不可用 → 音频为主 + 动作辅助
  场景3：不在区域内 → 纯音频路径（audio_only_cry_threshold=0.3）
```

---

### 4. 口鼻遮挡检测 (Occlusion Detection)

**核心逻辑**: `face_detector.detect_occlusion()`

```
三层抑制误报：
  1. HSV 肤色比例（H:[0,30] S:[10,255] V:[40,255] 加宽范围）
  2. FaceMesh 质量门控（≥406关键点 → 非皮肤×0.45；≥310 → ×0.65）
  3. 手部交叉验证（无手部重叠 → 置信度×0.6）

降级路径：
  Face Mesh 可用 → 精准纹理分析 + 手部交叉验证
  Face Mesh 不可用 → fallback 肤色+边缘+气道ROI
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

使用独立的 `face_mesh_absence_start_time` 计时器，与面部可见性追踪解耦。

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

排除：
  外部人手靠近不算踢被子
  Moro 惊跳反射时大幅衰减（×0.35）
```

---

### 7. 区域检测 (Region Detection)

**支持格式**:
- 旧版：矩形 2个对角点
- 新版：多边形 N个顶点（默认4点，婴儿床四边形）

**判断逻辑**:
```
6级优先级判断（从强到弱）：
  1. face_center_in_region → 金标准，直接判定在区域内
  2. face_bbox + face_center_in_region → 盖被子场景
  3. head_center_in_region + body_overlap ≥ 0.20 → 姿态辅助
  4. torso_overlap ≥ 0.50 + body_overlap ≥ 0.40 → 标准阈值
  5. torso_center_in_region + body_overlap ≥ 0.25 → 宽松兜底
  6. 无检测信号 → 保持上一帧状态，防止抖动

离开确认：body_overlap < 0.40 且 6帧滑窗中 50% 离开 → region_exit
进入确认：body_overlap ≥ 0.40 + 6帧滑窗 ≤ 50% 离开 → region_enter

边缘触发：
  只有状态翻转 (in→out / out→in) 才发通知
  不是持续在区域内或外就一直通知

预缓存抓拍：
  状态刚变化时立即存帧
  防抖确认后用缓存帧通知，确保画面与事件瞬间同步
```

---

### 8. 面部缺失检测 (Face Not Visible)

当人脸检测不到且姿态也无法确认头部位置，持续 ≥ `face_absence_duration_threshold`（默认 10s）触发 `face_not_visible` 告警。

---

## 事件存储与维测

**模块**: storage.py + SQLite

```
events 表（告警事件）：
  id, timestamp, type, level, confidence, photo_path, details

events_debug 表（诊断快照）：
  id, timestamp, snapshot_json (完整系统状态)
  每 30 帧写入一次，保留 48 小时
  支持按时间范围回溯分析历史误报/漏报

自动清理：
  event_retention_days: 30 天
  max_photo_size_mb: 1024 MB
```

**诊断快照内容**：presence、音频特征、哭声置信度、遮挡分数、曝光比例、区域状态、趴睡标志、面部可见性、活跃告警队列、系统温度。

---

## 双摄像头系统

### 硬件配置
- cam0: imx708_wide（常规 RGB）
- cam1: imx708_wide_noir（夜视，去 IR 滤光）

### 自动切换流程
```
每 30 帧 (~2秒) 检测一次
  ↓
测量中心 50% 区域亮度
  ↓
亮度 < 0.15 → dark_counter++
亮度 > 0.30 → bright_counter++
  ↓
dark_counter ≥ stable_frames(7) → 切换到夜视 (cam1)
bright_counter ≥ stable_frames(7) → 切回常规 (cam0)
  ↓
切换后重置计数器，300s 内不再切换
```

### 手动切换
发送 `SIGUSR2` 信号给 camera_server 进程触发强制切换。

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

## 容错与自愈

| 机制 | 说明 |
|------|------|
| 进程自动重启 | max_restart_attempts=5次，退避2,4,6,8,10秒 |
| 进程隔离 | 摄像头崩溃不影响推理，反之亦然 |
| 检测结果过期 | detection_stale > 阈值 → 停止显示旧状态，避免误导 |
| 存在确认缓冲 | 所有告警前置 presence 确认，空床不报 |
| 结果防抖 | 所有判断都有时序窗口，不依赖单帧 |
| 信号重入保护 | 信号处理函数禁止 `print()`，避免 stdout 重入 RuntimeError |
| in_region 滞后 | 检测块使用上一帧的 in_region 值（1帧滞后 @3fps ≈ 333ms），区域检测在末尾更新 |
| 推理线程自愈 | 推理线程崩溃自动重启并清除过期结果 |
| 优雅退出 | `shutting_down` 标志位防止退出时触发自动重启 |

---

## 配置参数总览

| 模块 | 关键参数 | 默认值 |
|------|----------|--------|
| 存在确认 | presence_window_size | 8帧 |
| | presence_confirm_ratio | 75% |
| 哭闹 | cry_confidence_threshold | 0.5 |
| | cry_duration_threshold | 2.0秒 |
| 遮挡 | occlusion_threshold | 0.75 |
| | occlusion_confirm_frames | 3帧（FaceMesh质量门控+手部交叉验证） |
| 裸露 | exposure_threshold | 0.35 |
| | exposure_duration_threshold | 8.0秒 |
| 区域 | region_body_overlap_threshold | 0.40 |
| | region_exit_window_size | 6帧 |
| | region_exit_confirm_ratio | 0.5 |
| 趴睡 | prone_duration_threshold | 5.0秒 |
| 面部缺失 | face_absence_duration_threshold | 10.0秒 |
| 冷却 | alert_cooldown_s | 10秒 |
| 双摄 | night_threshold | 0.15 |
| | day_threshold | 0.30 |
| | stable_frames | 7 |
| | min_switch_interval | 300秒 |

---

## 代码统计

| 文件 | 行数 | 职责 |
|------|------|------|
| src/supervision.py | 1852 | 监督核心，所有检测逻辑 |
| src/audio_gateway.py | 596 | 音频网关 + 哭声检测 |
| src/vision/face_detector.py | 636 | 人脸+表情+手部+遮挡 |
| src/preview_renderer.py | 497 | UI渲染 + 颜色编码 |
| inference_client.py | 487 | 三流解耦推理客户端 |
| src/audio_detector.py | 449 | 音频信号处理 |
| src/storage.py | 409 | SQLite存储 + 诊断快照 |
| src/notifier.py | 393 | 飞书通知 |
| main.py | 259 | 双进程管理 + 自动重启 |
| src/vision/body_detector.py | 259 | 姿态检测 |
| camera_server.py | 174 | 摄像头采集 + 双摄路由 |
| src/camera/dual_camera_proxy.py | 162 | 双摄互斥路由 |
| src/vision/region_detector.py | 158 | 区域判断 |
| src/camera/ambient_light_detector.py | 89 | 环境光线检测 |
| src/config.py | 54 | 配置管理 |
| src/camera/__init__.py | 3 | 包导出 |
| **总计** | **6477** | |
