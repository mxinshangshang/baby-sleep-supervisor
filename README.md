# Baby Sleep Supervisor - 婴幼儿睡眠监督系统

基于树莓派 5 的本地化婴幼儿睡眠监督系统，通过单摄像头实现睡眠状态监测、异常行为检测和实时通知。

---

## 项目概述

Baby Sleep Supervisor 面向婴幼儿睡眠场景，强调三点：

1. 非侵入式监督：只依赖摄像头和麦克风，不需要佩戴任何设备
2. 本地处理：全部在树莓派本地完成，保护隐私，不依赖网络
3. 稳定可信：优先保证长期运行、低误报和及时提醒

当前版本针对树莓派 5 + Camera Module 3 Wide 摄像头进行了优化，专注于0-6岁婴幼儿睡眠安全监护。

---

## 核心能力

| 功能 | 说明 |
|------|------|
| 哭闹检测 | 音频频谱 + 面部表情 + 肢体动作多模态融合检测哭闹 |
| 口鼻遮挡检测 | 检测是否有被褥、玩具、手等遮挡口鼻，防止窒息风险 |
| 趴睡/面部朝下检测 | 头部可见但面部关键点持续不可见时告警，防止窒息 |
| 踢被子检测 | 检测腿部/躯干是否裸露在被子外面 |
| 区域检测 | 检测婴儿是否离开指定的安全睡眠区域 |
| 惊跳反射过滤 | 识别 Moro 反射，避免将其误判为哭闹或踢被子 |
| 异常抓拍 | 异常事件发生时自动抓拍照片并存档 |
| 飞书通知 | 通过飞书机器人实时发送异常通知和抓拍照片 |
| 双进程架构 | 摄像头采集与推理解耦，稳定性更高 |
| 温控降频 | 高温时自动降低推理负载，保证长期运行 |
| 自动重启 | 子进程异常退出后自动恢复 |
| 双运行模式 | 支持可视化预览模式和纯后台无头模式 |
| 事件存档 | 所有异常事件和抓拍照片本地持久化存储 |

---

## 推荐硬件

| 组件 | 推荐 | 说明 |
|------|------|------|
| 主板 | Raspberry Pi 5 8GB | 推荐 8GB，长时间运行更稳 |
| 摄像头 | Camera Module 3 Wide | 大广角适合监控婴儿床全景 |
| 存储 | SSD 512G | 大量抓拍照片存储需求 |
| 电源 | 官方 5V 5A | 避免供电不足 |
| 散热 | 主动或较强被动散热 | 长时间推理强烈建议配置 |
| 麦克风 | USB 麦克风 | 可选，用于声音辅助哭闹检测 |

---

## 部署模式

### 1. 婴儿床顶机位 (默认)

适合：
1. 摄像头安装在婴儿床正上方
2. 俯视角度，覆盖整个婴儿床区域

重点监督：
1. 口鼻遮挡
2. 踢被子/肢体裸露
3. 区域活动
4. 面部表情识别

### 2. 侧机位

适合：
1. 摄像头安装在婴儿床侧面
2. 水平角度，可以更好地观察面部表情和呼吸起伏

重点监督：
1. 哭闹表情
2. 呼吸状态
3. 身体移动

---

## 快速开始

### 1. 环境准备

当前树莓派运行方式复用现有环境：

- 摄像头采集进程使用系统 Python：`/usr/bin/python3`
- 推理进程复用 `kid_supervisor_v3` 已配置好的 Python 3.11 虚拟环境：`/home/mxin/.openclaw/workspace/kid_supervisor_v3/venv_311/bin/python`

因此正常运行不需要在 Baby 项目里重复安装 MediaPipe。

```bash
cd /home/mxin/.openclaw/workspace/baby_sleep_supervisor
```

### 2. 配置参数

编辑 `config.yaml`，根据实际安装位置调整检测区域和灵敏度参数。

### 3. 启动系统

```bash
# 带预览窗口模式
./start.sh

# 无头后台模式
./start_headless.sh
```

---

## 区域校准

首次部署建议进行区域校准，设置婴儿床的安全边界：

```bash
/usr/bin/python3 calibrate_region.py
```

步骤：
1. 标定工具会复用 `camera_server.py` 的同源画面；如果没有运行中的摄像头服务，会自动用 `/usr/bin/python3` 启动一个临时服务
2. 启动后默认不显示旧区域，直接等待新的四点输入
3. 依次左键点击婴儿床四个角点，第四个点后自动闭合并锁定区域
4. 如果想重画，继续点击第五个点会清空前四点并开始新一轮四点输入
5. 右键或 `u` 撤销未完成的上一个点，`r` 重置全部点
6. 按 `s` 保存安全区域，配置自动写回 `config.yaml`
7. 保存后在英文通知项选择框里勾选需要发送飞书通知的告警类型，按 `Enter`/`s` 保存到 `notification.enabled_alert_types`

---

## 项目结构

```text
baby_sleep_supervisor/
├── main.py                 # 主启动器，双进程管理
├── camera_server.py        # 摄像头采集服务（原生BGR，零拷贝传输）
├── inference_client.py     # 推理和业务逻辑客户端（多格式自适应解码）
├── calibrate_region.py     # 安全区域校准工具（多格式自适应解码）
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖列表
├── setup_venv.py           # 虚拟环境搭建脚本
├── start.sh                # 带预览启动脚本
├── start_headless.sh       # 无头模式启动脚本
├── src/
│   ├── config.py           # 配置加载模块
│   ├── supervision.py      # 监督逻辑核心（趴睡/区域/哭闹/遮挡/裸露）
│   ├── audio_detector.py   # 音频哭声频谱分析
│   ├── audio_gateway.py    # 多模态音频-视觉融合网关
│   ├── notifier.py         # 通知模块（飞书）
│   ├── preview_renderer.py # 预览窗口渲染（含检测过期指示）
│   ├── storage.py          # 数据存储模块
│   └── vision/
│       ├── face_detector.py    # 人脸/表情/口鼻遮挡检测
│       ├── body_detector.py    # 人体姿态/肢体裸露检测
│       └── region_detector.py  # 区域检测
├── data/
│   ├── photos/             # 异常事件抓拍照片
│   └── events.db           # 事件数据库
├── scripts/                # 辅助脚本
└── docs/                   # 文档目录
```

---

## 当前代码设计

### 双进程运行模型

系统由 `main.py` 统一管理两个子进程：

1. `camera_server.py` 使用系统 Python 启动，负责 Picamera2 摄像头初始化、采集、原生 BGR 帧 TCP 发送（零 JPEG 编解码，客户端自适应解码）。
2. `inference_client.py` 使用 `kid_supervisor_v3` 的 Python 3.11 虚拟环境启动，负责接收帧（多格式自适应：BGR/YUV420/RGB888/pickle+JPEG）、运行 MediaPipe/OpenCV 推理、渲染预览和触发告警。

这样可以把摄像头驱动依赖和 AI 推理依赖隔离开：摄像头继续使用 Raspberry Pi OS 原生 `picamera2` 环境，推理继续复用已经验证可用的 MediaPipe 环境。

### 主进程守护与退出策略

`main.py` 会监控摄像头进程和推理进程的状态：

- 子进程异常退出后按配置自动重启。
- 稳定运行超过 `restart_reset_after_s` 后清零重启计数。
- 超过 `max_restart_attempts` 后执行安全退出。
- `SIGINT` / `SIGTERM` 会先停止推理进程，再停止摄像头进程，最后销毁 OpenCV 窗口。

预览模式下 `q` 键由推理进程向主进程发送 `SIGUSR1`，主进程按以下顺序安全退出：

1. 先终止推理进程，再终止摄像头进程。
2. 销毁残留 OpenCV 窗口。
3. 确保 Camera Module 3 Wide 占用释放。

不支持通过 `q` 重启程序；如需重启，退出后重新执行 `./start.sh`。

### 检测流水线

`SleepSupervisor` 是监督核心，按帧执行以下逻辑：

1. 姿态检测：作为人体存在和区域判断的主链路，支持侧脸、背脸、局部身体可见等睡姿。
2. 人脸检测和 Face Mesh：作为存在性增强证据，并在 Face Mesh 可靠时用于哭闹和口鼻遮挡判断。
3. Presence 确认：综合姿态关键点、头/躯干锚点、人脸框和分割掩码，连续多帧确认画面中确实有婴儿。
4. 时间窗口平滑：对 presence、哭闹置信度、遮挡置信度、裸露比例做滑动平均，降低瞬时误报。
5. 持续时间判断：异常必须持续超过配置阈值才保存事件和发送告警。
6. 冷却控制：同类事件在冷却时间内不会重复通知。

异常事件会保存到 SQLite，并把抓拍照片写入 `data/photos/`。

### 安全区域设计

区域检测支持两种配置格式：

```yaml
# 旧版矩形，两点对角线
safe_region: [[50, 50], [590, 430]]

# 新版多边形，推荐四点区域
safe_region:
  - [126, 95]
  - [122, 364]
  - [538, 378]
  - [536, 88]
```

`RegionDetector` 会根据配置自动识别矩形或多边形。多边形模式使用 `cv2.pointPolygonTest` 判断中心点是否在区域内，并用身体边界框的角点和中心点估算与安全区域的重叠比例。满足中心点在区域内或重叠比例超过 70% 时认为仍在安全区域内。

区域告警还会先判断是否存在有效人体：系统使用姿态关键点、头部/躯干锚点、人脸框和连续帧 presence 分数确认画面中确实有婴儿。空床或 MediaPipe 偶发误识别时不会触发离开区域告警。

区域判断优先使用躯干和头部，而不是单纯依赖全身 bbox：如果躯干中心、躯干重叠、头部中心或身体主体仍在 Safe Region 内，就不会因为单个手脚伸出区域而误报离区。

侧脸/背脸睡觉时，人脸可能没有 Face Mesh，但姿态和头/躯干仍可确认 presence 和区域状态；“脸不可见”不会被当成口鼻遮挡。

### 预览界面设计

预览界面使用 OpenCV `putText` 渲染，因此当前 UI 文案统一使用英文和 ASCII，避免中文或 emoji 在 OpenCV 窗口中显示成 `?`。

预览窗口支持：

- 检测状态叠加：Cry、Occlusion、Exposure、In Region。
- 安全区域叠加：矩形或四点多边形、半透明填充和顶点标记。
- 快捷键开关：帮助、检测框、安全区域、统计信息。
- 最近事件临时显示。

### 飞书通知链路

通知模块优先复用已经打通的 OpenClaw 飞书 App 通道：

1. 运行时只读 `/home/mxin/.openclaw/openclaw.json`，读取已有 Feishu App 配置。
2. 只读 OpenClaw session 索引，找到最近的飞书 `open_id`。
3. 通过 Feishu tenant access token 发送文本消息。
4. 对异常抓拍图片先调用 Feishu 图片上传接口获取 `image_key`，再发送图片消息。
5. 如果 OpenClaw 通道不可用，则回退到 `config.yaml` 中配置的 webhook 发送方式。

该设计不会修改 OpenClaw 自身配置或功能，只是 Baby 项目在运行时读取并复用已有通道。

### 模块职责

| 模块 | 职责 | 关键点 |
|------|------|--------|
| `main.py` | 主启动器和守护进程 | 管理摄像头/推理两个子进程，处理自动重启、安全退出和 `q` 键信号 |
| `camera_server.py` | 摄像头采集服务 | 使用 Picamera2 `capture_array()` 获取原生BGR帧，通过 TCP 发送原始像素数据（零JPEG编解码） |
| `inference_client.py` | 推理客户端 | 连接摄像头服务，多格式自适应解码（BGR/YUV420/RGB888/pickle+JPEG），调用监督器，渲染预览，处理快捷键 |
| `calibrate_region.py` | 安全区域标定工具 | 在摄像头预览中手动点击四个角点，保存到 `config.yaml`，支持多格式帧解码 |
| `src/config.py` | 配置加载 | 读取 YAML 配置并准备数据目录 |
| `src/supervision.py` | 监督核心 | 串联人脸、姿态、区域、音频、存储和通知逻辑 |
| `src/audio_detector.py` | 音频检测 | 哭声频谱分析：RMS音量、基频峰值、频谱质心、高频能量占比 |
| `src/audio_gateway.py` | 音频网关 | 独立进程采集音频，多模态融合（音频+视觉+动作），永远不阻塞主推理进程 |
| `src/notifier.py` | 通知模块 | 发送控制台、飞书文本和飞书图片通知，优先复用 OpenClaw 通道 |
| `src/storage.py` | 本地存储 | 保存异常照片和 SQLite 事件记录 |
| `src/preview_renderer.py` | 预览渲染 | 绘制检测状态、安全区域、事件提示，检测过期时隐藏旧结果并显示警告 |
| `src/vision/face_detector.py` | 人脸检测 | MediaPipe 人脸检测、Face Mesh、哭闹表情和口鼻遮挡特征 |
| `src/vision/body_detector.py` | 姿态检测 | MediaPipe Pose、身体关键点和肢体裸露估算 |
| `src/vision/region_detector.py` | 区域检测 | 矩形/多边形安全区域判断和区域绘制 |

### 端到端数据流

```text
Camera Module 3 Wide
        |
        v
camera_server.py
  Picamera2 capture_array() → 原生 BGR
  原始像素数据 TCP 发送（零编解码）
        |
        v
inference_client.py
  LatestFrameReceiver: TCP 接收，多格式自适应解码
  LatestInferenceWorker: SleepSupervisor.process_frame()
        |
        +--> FaceDetector: 人脸、表情、口鼻遮挡
        +--> BodyDetector: 姿态、肢体裸露
        +--> RegionDetector: 是否在安全区域内
        +--> AudioGateway: 音频哭声检测 + 多模态融合
        |
        v
事件判断
  平滑窗口
  持续时间阈值
  冷却时间
        |
        +--> Storage: 保存照片和事件数据库
        +--> Notifier: 控制台/飞书文本/飞书图片
        +--> PreviewRenderer: 预览窗口叠加显示（含检测过期提示）
```

摄像头进程和推理进程之间只传输图像帧，不直接共享摄像头对象。推理进程异常时不会直接持有摄像头设备，主进程可以更可靠地释放并重启摄像头服务。

### 异常事件生命周期

一次异常从画面变化到最终通知，经过以下阶段：

1. **单帧检测**：从当前帧提取人脸、姿态、皮肤区域、安全区域等特征。
2. **置信度计算**：得到哭闹置信度、遮挡置信度、裸露比例或区域状态。
3. **滑动平滑**：将最近多帧结果放入窗口求均值，减少单帧噪声。
4. **持续时间确认**：异常状态首次出现时记录开始时间，持续超过阈值才进入告警。
5. **冷却判断**：同类事件在冷却窗口内只保留状态，不重复发送通知。
6. **事件落盘**：保存抓拍照片，写入 SQLite 事件记录，生成事件 ID。
7. **通知发送**：构造文本内容，发送飞书文本；如果有照片且配置允许，再发送图片。
8. **预览反馈**：在窗口中显示当前异常状态和最近事件。

该生命周期用于降低误报：单帧识别结果不会直接触发通知，必须经过平滑、持续时间和冷却控制。

### 检测策略细节

| 事件 | 输入来源 | 主要判断 | 默认级别 |
|------|----------|----------|----------|
| 哭闹 | 音频频谱 + Face Mesh + Pose 动作 | 多模态融合（音频50%+视觉35%+动作15%），三重交叉确认 | warning / danger |
| 口鼻遮挡 | Presence + Face Mesh + 手部检测 | FaceMesh 可用时检测口鼻 ROI 特征 + 手部重叠；不可用时用气道 ROI 兜底 | danger |
| 趴睡/面部朝下 | Presence + Pose head_bbox + Face Mesh | 头框可见但 FaceMesh 持续 >10s 不可用 → 怀疑面部朝下 | danger |
| 踢被子/肢体裸露 | Presence + Pose + 肤色检测 | 身体区域肤色占比 + 逐肢体分析；单手臂不告警；惊跳反射时衰减 | warning |
| 离开安全区域 | Presence + Pose + RegionDetector | 躯干/身体重叠比例 + 边缘触发通知（仅状态变化时通知一次） | warning |

当前实现偏向轻量级本地推理和启发式规则组合，避免在树莓派上运行过重模型导致长期运行稳定性下降。

#### 趴睡/面部朝下检测

针对 2-3 月龄 SIDS 高风险婴儿。仅当 **Pose 检测到头部框，但 FaceMesh 持续 10 秒以上无法获取面部关键点** 时触发 danger 告警。正常侧睡（面部朝向侧方、FaceMesh 可见）不会误触发。

#### 惊跳反射（Moro Reflex）过滤

2-3 月龄婴儿仍存在惊跳反射，双臂对称外展会瞬时提高动作躁动和肤色裸露比例。系统通过检测双腕移动向量的镜像对称性（夹角 >120°）来识别 Moro 反射，识别后对肢体动作信号衰减 70%，对裸露比例衰减 65%，避免误报。

### 飞书图片通知细节

异常事件带图片时，通知模块会按以下顺序处理：

1. 将本地抓拍图缩放到飞书限制以内。
2. 使用 OpenClaw Feishu App 凭据获取 `tenant_access_token`。
3. 调用飞书图片上传接口上传 JPEG，获取 `image_key`。
4. 使用 `image_key` 发送图片消息给最近 OpenClaw 飞书会话对应的 `open_id`。
5. 如果 OpenClaw App 通道失败，则尝试 webhook 回退逻辑。

OpenClaw 的配置和会话文件只读访问。Baby 项目不会写入 OpenClaw 配置，也不会改变 OpenClaw 自身消息收发行为。

### 本地存储设计

异常数据分为两类保存：

- 图片文件：保存到 `data/photos/` 下，文件名包含时间信息，便于人工查找。
- 事件记录：保存到 SQLite 数据库 `data/events.db`，记录事件类型、级别、消息、详情和图片路径。

存储层和通知层解耦：即使飞书发送失败，事件和照片仍会先保存在本地，便于事后排查。

---

## 配置说明

核心配置项说明：

```yaml
# 摄像头配置
camera:
  width: 640
  height: 480
  fps: 15
  format: YUV420
  use_full_sensor_fov: true

# 检测配置
detection:
  # 哭闹检测
  cry_detection_enabled: true
  cry_confidence_threshold: 0.7
  cry_duration_threshold: 2.0
  
  # 踢被子检测
  limb_exposure_enabled: true
  exposure_threshold: 0.3
  
  # 口鼻遮挡检测
  occlusion_detection_enabled: true
  occlusion_threshold: 0.6
  
  # 区域检测，支持两点矩形或四点/多点多边形
  region_detection_enabled: true
  safe_region:
    - [126, 95]
    - [122, 364]
    - [538, 378]
    - [536, 88]

# 通知配置
notification:
  feishu_enabled: true
  feishu_webhook: "your-webhook-url"  # OpenClaw 通道不可用时的回退 webhook
  capture_photo_on_alert: true
  enabled_alert_types:
    - cry_detected          # Crying
    - occlusion_detected    # Face covered
    - limb_exposure         # Left hand exposed
    - region_exit           # Out of safe region

supervision:
  alert_cooldown_s: 60.0

# 存储配置
storage:
  photo_retention_days: 30
  sqlite_path: data/events.db
```

---

## 事件级别

系统定义了三个级别的异常事件：

| 级别 | 说明 | 响应 |
|------|------|------|
| 注意 (Notice) | 轻微异常，如短暂肢体移动 | 记录日志，不通知 |
| 警告 (Warning) | 中度异常，如短暂踢被子 | 记录日志，连续超过阈值则通知 |
| 危险 (Danger) | 严重异常，如口鼻遮挡、哭闹 | 立即抓拍并发送通知 |

---

## 当前实现边界

当前版本优先保证可靠性、可运行性和低误报，以下是当前实现边界：

### 已实现并作为主链路使用

1. 单摄像头本地监护。
2. 摄像头采集和 AI 推理双进程隔离。
3. 原生 BGR 零编解码传输，多格式自适应解码（BGR/YUV420/RGB888/pickle+JPEG）。
4. 强制完整传感器阵列下采样，最大化广角。
5. MediaPipe 人脸、Face Mesh、Pose 推理。
6. 哭闹表情启发式检测 + 多模态融合（音频+视觉+动作）。
7. 音频哭声频谱分析（独立进程，不阻塞主推理）。
8. 口鼻遮挡启发式检测。
9. 趴睡/面部朝下检测（2-3月龄 SIDS 高风险）。
10. 惊跳反射（Moro Reflex）过滤。
11. 肢体裸露/踢被子启发式检测。
12. 矩形或四点/多点多边形安全区域检测（边缘触发通知）。
13. 异常事件抓拍和 SQLite 事件记录。
14. OpenClaw 飞书通道复用，支持异常文本和图片通知。
15. 预览模式和无头后台模式。
16. 子进程异常自动重启和 `q` 键安全重启/退出。
17. 推理工作线程崩溃自动重启，过期检测结果清除。
18. 预览界面检测过期提示（结果>2s时隐藏并显示警告）。

### 配置存在但不是当前主链路

1. 呼吸异常相关配置存在，但当前版本没有把呼吸检测作为稳定主功能。
2. 温控配置存在，后续可以继续接入动态调节推理帧率和模型复杂度。

### 当前技术限制

1. 哭闹、遮挡、踢被子检测使用轻量级启发式规则，不等同于医疗级判断。
2. 多边形区域重叠判断采用角点和中心点估算，不是像素级精确 IoU。
3. OpenClaw 飞书接收人当前通过最近会话自动发现，如果最近会话不是目标接收人，可能需要后续改为显式配置。
4. OpenCV 预览窗口使用 `putText`，中文和 emoji 可能显示为 `?`，因此 UI 文案使用英文。
5. 系统假设同一时间只有一个摄像头服务占用 Camera Module 3 Wide。
6. 当前不做云端同步，所有图片和事件默认只保存在本机。

### 暂不实现

1. 云端同步。
2. 语音对讲。
3. 多摄像头协同。
4. 医疗级呼吸监测。
5. 重量级目标检测模型常驻运行。

### 后续可改进方向

1. 将飞书 `receive_id` 改为 Baby 项目显式配置，减少对最近 OpenClaw 会话的依赖。
2. 缓存 Feishu `tenant_access_token`，减少频繁请求 token。
3. 为异常检测增加离线回放工具，用已保存照片或视频调参。
4. 增加更细的事件统计页面，例如每晚异常次数、持续时间和趋势。
5. 根据树莓派温度动态降低推理帧率或模型复杂度。
6. 将声音哭闹检测接入主事件链路，作为视觉哭闹检测的辅助证据。

---

## 许可证

仅供家庭非商业使用。
