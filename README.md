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
| 哭闹检测 | 通过面部表情和声音检测婴儿是否哭闹 |
| 踢被子检测 | 检测是否有肢体裸露在被子外面 |
| 口鼻遮挡检测 | 检测是否有被褥、玩具等遮挡口鼻，防止窒息风险 |
| 区域检测 | 检测婴儿是否离开指定的安全睡眠区域 |
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

```bash
cd /home/mxin/.openclaw/workspace/baby_sleep_supervisor
python3 setup_venv.py
```

### 2. 配置参数

编辑 `config.yaml`，根据实际安装位置调整检测区域和灵敏度参数。

### 3. 启动系统

```bash
# 带预览窗口模式
/usr/bin/python3 main.py

# 无头后台模式
/usr/bin/python3 main.py --no-preview
```

---

## 区域校准

首次部署建议进行区域校准，设置婴儿床的安全边界：

```bash
/usr/bin/python3 calibrate_region.py
```

步骤：
1. 在预览窗口中用鼠标框选婴儿床的安全区域
2. 按 `s` 保存配置
3. 配置自动写回 `config.yaml`

---

## 项目结构

```text
baby_sleep_supervisor/
├── main.py                 # 主启动器，双进程管理
├── camera_server.py        # 摄像头采集服务
├── inference_client.py     # 推理和业务逻辑客户端
├── calibrate_region.py     # 安全区域校准工具
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖列表
├── setup_venv.py           # 虚拟环境搭建脚本
├── start.sh                # 带预览启动脚本
├── start_headless.sh       # 无头模式启动脚本
├── src/
│   ├── config.py           # 配置加载模块
│   ├── supervision.py      # 监督逻辑核心
│   ├── notifier.py         # 通知模块（飞书）
│   ├── preview_renderer.py # 预览窗口渲染
│   ├── storage.py          # 数据存储模块
│   └── vision/
│       ├── face_detector.py    # 人脸/表情检测
│       ├── body_detector.py    # 人体/肢体检测
│       ├── occlusion_detector.py  # 遮挡检测
│       └── region_detector.py  # 区域检测
├── data/
│   ├── photos/             # 异常事件抓拍照片
│   └── events.db           # 事件数据库
└── docs/                   # 文档目录
```

---

## 配置说明

核心配置项说明：

```yaml
# 摄像头配置
camera:
  width: 640
  height: 480
  fps: 15
  jpeg_quality: 80

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
  
  # 区域检测
  region_detection_enabled: true
  safe_region: [[50, 50], [590, 430]]

# 通知配置
notification:
  feishu_enabled: true
  feishu_webhook: "your-webhook-url"
  alert_cooldown_s: 60.0
  capture_photo_on_alert: true

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

当前版本优先保证可靠性和低误报，暂不实现：
1. 云端同步
2. 语音对讲
3. 多摄像头协同
4. 复杂AI模型

---

## 许可证

仅供家庭非商业使用。
