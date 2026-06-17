# Baby Sleep Supervisor - 婴幼儿睡眠监督系统

基于树莓派 5 的本地化婴幼儿睡眠监督系统，通过单摄像头实现睡眠状态监测、异常行为检测和实时通知。

---

## 📌 项目概述

Baby Sleep Supervisor 面向婴幼儿睡眠场景，强调三点核心原则：

1. **非侵入式监督**：只依赖摄像头和麦克风，不需要佩戴任何设备
2. **本地处理**：全部在树莓派本地完成，保护隐私，不依赖网络
3. **稳定可信**：优先保证长期运行、低误报和及时提醒

当前版本针对 **树莓派 5 + Camera Module 3 Wide** 深度优化，专注于0-6岁婴幼儿睡眠安全监护。

---

## ✨ 核心能力 v1.0

| 功能 | 实现方式 | 说明 |
|------|----------|------|
| 🎭 **哭闹检测** | 音频频谱 + 面部表情 + 肢体动作 | **四证据链多模态融合**，含Moro惊跳过滤 |
| 😷 **口鼻遮挡检测** | Face Mesh纹理分析 + 手部重叠 | 防止被褥/手/玩具窒息风险 |
| 🛌 **趴睡检测** | Face Mesh缺失 + 姿态拓扑推断 | 面部持续不可见8秒告警 |
| 🦵 **踢被子检测** | 身体关键点可见性分析 | 下肢/躯干裸露持续8秒告警 |
| 🚧 **区域检测** | 多边形重叠判断 | 婴儿离开安全睡眠区域告警 |
| 👤 **存在确认** | 人脸+姿态联合判定 | 所有告警前置确认，空床不报 |
| 🔇 **面部缺失检测** | 头框存在但无Face Mesh | 头被盖住/侧脸朝下告警 |
| 📸 **异常抓拍** | 事件预缓存机制 | 告警画面与事件瞬间精确同步 |
| 📱 **飞书通知** | Webhook机器人 | 实时告警+抓拍照片，支持告警类型选择 |
| 🌡️ **温控降频** | CPU温度监控 | 65°C降推理帧率，75°C降模型复杂度 |
| 🔄 **双进程架构** | TCP:65433解耦 | 摄像头崩溃不影响推理，自动重启 |
| 🖥️ **双运行模式** | 预览/无头 | 桌面可视化或纯后台服务 |
| 💾 **事件存档** | SQLite本地持久化 | 30天自动清理，1GB照片上限 |

---

## 🛠️ 推荐硬件

| 组件 | 推荐 | 说明 |
|------|------|------|
| 主板 | Raspberry Pi 5 8GB | 推荐 8GB，长时间运行更稳定 |
| 摄像头 | Camera Module 3 Wide | 大广角适合监控婴儿床全景 |
| 存储 | SSD 512G | 大量抓拍照片存储需求 |
| 电源 | 官方 5V 5A | 避免供电不足导致的不稳定 |
| 散热 | 主动/较强被动散热 | 长时间推理强烈建议配置 |
| 麦克风 | USB 麦克风 | 可选，音频辅助哭闹检测 |

---

## 🎯 部署模式

### 婴儿床顶机位 (默认推荐)

```
    [摄像头]
        │
        ▼ 俯视90°
┌───────────────┐
│   婴儿床      │
│   ○ 宝宝      │
└───────────────┘
```

**优势**:
- 完整覆盖婴儿床区域
- 身体关键点检测最准确
- 踢被子/区域检测效果最佳

### 侧机位

```
              [摄像头]
                  │
                  ▼ 水平视角
┌───────────────┐
│   ○ → 宝宝     │
└───────────────┘
```

**优势**:
- 面部表情识别更准确
- 哭闹检测效果更好
- 适合配合顶机位双摄

---

## 🚀 快速开始

### 1. 环境说明

项目采用**双Python环境**架构，零重复安装：

| 进程 | Python环境 | 说明 |
|------|-----------|------|
| 摄像头采集 | 系统 Python `/usr/bin/python3` | 原生 picamera2 支持 |
| 算法推理 | Python 3.11 虚拟环境 | 复用 `kid_supervisor_v3` 已配置环境 |

**不需要在 Baby 项目里重复安装 MediaPipe**。

```bash
cd /home/mxin/.openclaw/workspace/baby_sleep_supervisor
```

### 2. 首次配置

编辑 `config.yaml` 核心参数：

```yaml
# 1. 安全区域 - 先运行校准工具获得准确坐标
detection:
  safe_region: [[442, 202], [913, 60], [893, 530], [429, 422]]

# 2. 飞书通知 - 填入你的webhook
notification:
  feishu_enabled: true
  feishu_webhook: https://open.feishu.cn/open-apis/bot/v2/hook/your-token
  enabled_alert_types: [cry_detected, occlusion_detected, region_exit]
```

### 3. 区域校准 (必须)

```bash
/usr/bin/python3 calibrate_region.py
```

**操作步骤**:
1. 自动启动临时摄像头服务
2. 依次**左键点击**婴儿床四个角点（顺时针/逆时针均可）
3. 点击第四点后自动绘制多边形区域
4. 点击第五点 → 清除重来，支持无限轮校准
5. `u` 撤销上一点，`r` 重置全部
6. 按 `s` 保存 → 配置自动写回 `config.yaml`
7. 最后选择需要通知的告警类型，`Enter` 保存

### 4. 启动系统

```bash
# 带预览窗口模式（桌面调试）
./start.sh

# 无头后台模式（生产部署）
./start_headless.sh

# 停止
pkill -f "python3.*main.py"
```

---

## 🏗️ 系统架构

### 三流解耦设计 (核心创新)

```
┌───────────────────────────────────────────────────────────┐
│                    三线程并行解耦                          │
├──────────────────┬──────────────────┬───────────────────┤
│   接收线程       │   算法线程        │   UI线程          │
│   15fps 恒定     │   3fps 可控       │   15fps 恒定      │
│   只保留最新帧   │   高温自动降频    │   永远流畅        │
└──────────────────┴──────────────────┴───────────────────┘
```

**关键原则**: 算法变慢不影响预览流畅度，摄像头采集永远不受推理影响。

详细架构文档 → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 🎨 预览界面说明

### 颜色编码系统

| 颜色 | 元素 | 算法来源 | 说明 |
|------|------|----------|------|
| 🔵 蓝 | 人脸框 | MediaPipe Face Detection | 检测到的婴儿脸部 |
| 🟠 橙 | 头部框 | MediaPipe Pose 0-10 | 姿态拟合的头部范围 |
| 🟣 紫 | 躯干框 | MediaPipe Pose 11,12,23,24 | 肩+髋拟合的身体范围 |
| 🟡 黄 | 附近的手 | MediaPipe Hands | 手在头部附近 |
| 🔴 红 | 风险手 | MediaPipe Hands + 重叠判断 | 手遮挡口鼻，高风险 |
| 🟢 绿 | 安全区域 | 手工标定 | 婴儿床边界 |
| ⚪ 青 | 姿态骨架 | MediaPipe Pose | 33个关键点 + 连线 |

### 快捷键

| 按键 | 功能 |
|------|------|
| `q` | 退出系统 |
| `h` | 切换帮助显示 |
| `d` | 切换检测框显示 |
| `r` | 切换安全区域显示 |
| `s` | 切换统计信息 |

### 状态指示

```
左上角：
  Presence: Yes/Uncertain/No  - 婴儿存在确认
  Face: Front/Side/Mesh only  - 面部可见状态
  Cry: 0.XX [A/V]             - 哭闹置信度 [声学分/视觉分]
  Occlusion: 0.XX             - 遮挡置信度
  Coverage: 0.XX [level]      - 裸露比率
  Region: in/out_region       - 区域状态

右上角：
  FPS                         - 预览帧率
  NORMAL/WARNING/DANGER       - 系统状态
```

---

## 🔍 检测原理详解

### 😭 哭闹检测 - 四证据链融合

```
第1层 面部：嘴巴张开持续度 + 开合节律
第2层 肢体：头部摆动 + 手臂躁动 + 全身运动
第3层 音频：50Hz陷波降噪 + 哭声频段(250-1500Hz)频谱
第4层 节律：哭声周期性爆发（0.4-1.3秒/周期，≥3次连续）

→ 多模态加权融合 → 持续2秒以上确认告警
```

### 🦵 踢被子检测 - 无监督思路

```
核心洞察：
  盖被子 → 皮肤被遮挡 → MediaPipe 检测不到高置信度关键点
  踢被子 → 皮肤裸露 → MediaPipe 清晰检测到腿/手臂关键点

算法：
  下肢4个关键点(25-28) + 上肢6个关键点(11-16)
  → 可见比率 > 35%
  → 持续8秒 → 告警
```

详细科学依据 → [docs/SCIENTIFIC_VALIDATION.md](docs/SCIENTIFIC_VALIDATION.md)

---

## ⚙️ 关键配置调优

### 灵敏度调节

```yaml
detection:
  # 哭闹 - 越小孩越灵敏
  cry_confidence_threshold: 0.5    # 建议 0.4-0.7
  cry_duration_threshold: 2.0      # 建议 1.5-3.0 秒

  # 遮挡 - 越小越灵敏
  occlusion_threshold: 0.6         # 建议 0.5-0.8
  occlusion_duration_threshold: 1.0

  # 裸露 - 越小越灵敏
  exposure_threshold: 0.35         # 建议 0.25-0.5
  exposure_duration_threshold: 8.0

  # 区域 - 重叠比率阈值
  region_body_overlap_threshold: 0.55  # 越小越容易判"离开"
```

### 温控策略

```yaml
thermal:
  enabled: true
  temp_warn_c: 65.0        # ≥65°C 降推理帧率
  temp_throttle_c: 75.0    # ≥75°C 降模型复杂度到Lite
  throttle_inference_fps: 2
```

---

## 📂 项目结构

```text
baby_sleep_supervisor/
├── main.py                      # 双进程主启动器 + 自动重启管理
├── camera_server.py             # 摄像头采集服务 (BGR零拷贝)
├── inference_client.py          # 推理客户端 - 三流解耦核心
├── calibrate_region.py          # 安全区域校准工具
├── config.yaml                  # 所有配置集中管理
├── start.sh                     # 带预览启动
├── start_headless.sh            # 无头模式启动
├── src/
│   ├── supervision.py          # 1513行 - 监督核心，所有检测逻辑
│   ├── audio_gateway.py        # 音频网关 - 独立进程哭声检测
│   ├── audio_detector.py       # 音频信号处理 - 频谱+节律
│   ├── vision/
│   │   ├── face_detector.py    # 人脸+表情+手部+遮挡检测
│   │   ├── body_detector.py    # MediaPipe Pose 姿态检测
│   │   └── region_detector.py  # 多边形区域判断
│   ├── preview_renderer.py     # 预览渲染 - 颜色编码系统
│   ├── notifier.py             # 飞书通知模块
│   ├── storage.py              # SQLite 事件存储
│   └── config.py               # 配置加载
├── docs/
│   ├── ARCHITECTURE.md         # 完整系统架构文档
│   ├── SCIENTIFIC_VALIDATION.md
│   ├── AUDIO_CRY_DETECTION.md
│   ├── RECENT_DESIGN_UPDATES.md
│   └── QUICKSTART.md
└── data/
    ├── events.db               # SQLite 事件数据库
    └── photos/                 # 告警抓拍存档
```

---

## 📊 代码统计 (v1.0)

| 模块 | 代码行数 | 占比 |
|------|----------|------|
| 监督核心 supervision.py | 1513 | 31% |
| 视觉检测 face_detector.py | 631 | 13% |
| 音频网关 audio_gateway.py | 581 | 12% |
| 预览渲染 preview_renderer.py | 457 | 9% |
| 音频检测 audio_detector.py | 449 | 9% |
| 通知模块 notifier.py | 393 | 8% |
| 存储模块 storage.py | 387 | 8% |
| 姿态检测 body_detector.py | 204 | 4% |
| 区域检测 region_detector.py | 158 | 3% |
| 配置 config.py | 54 | 1% |
| **总计** | **4846** | **100%** |

---

## 📚 相关文档

| 文档 | 内容 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 🔥 完整系统架构 - 必看 |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 快速上手指南 |
| [docs/SCIENTIFIC_VALIDATION.md](docs/SCIENTIFIC_VALIDATION.md) | 检测算法科学依据 |
| [docs/AUDIO_CRY_DETECTION.md](docs/AUDIO_CRY_DETECTION.md) | 音频哭闹检测技术细节 |
| [docs/RECENT_DESIGN_UPDATES.md](docs/RECENT_DESIGN_UPDATES.md) | 设计迭代历史 |

---

## ⚠️ 重要说明

1. **本系统为辅助监护工具，不能替代成人看护**
2. 所有检测基于计算机视觉，存在误报和漏报可能
3. 定期检查系统运行状态，不要完全依赖自动告警
4. 建议首次部署后连续观察24小时，根据实际情况调参

---

## 🔄 版本信息

- **Version**: 1.0.0
- **Last Updated**: 2026-06-16
- **Target Platform**: Raspberry Pi 5 + Camera Module 3 Wide
