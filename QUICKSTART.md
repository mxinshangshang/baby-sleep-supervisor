# 快速开始指南

## 1. 系统要求
- 硬件: Raspberry Pi 5 (4GB/8GB) + Camera Module 3 Wide + 5V 5A 电源
- 系统: Raspberry Pi OS (Bookworm 64位)
- 存储: 至少16GB SD卡或SSD，建议使用SSD存储抓拍照片

## 2. 安装步骤

### 2.1 启用摄像头
```bash
sudo raspi-config
# 选择 Interface Options -> Camera -> Enable
# 重启树莓派
```

### 2.2 运行环境
```bash
cd /home/mxin/.openclaw/workspace/baby_sleep_supervisor
```

当前项目采用双 Python 运行方式：

- 摄像头采集：系统 Python `/usr/bin/python3`，用于调用 Raspberry Pi 原生 `picamera2`。
- 推理检测：复用 `kid_supervisor_v3/venv_311/bin/python`，用于调用已安装验证过的 MediaPipe。

正常运行不需要在 Baby 项目里重复安装 MediaPipe。

### 2.3 配置飞书通知
当前优先复用已经打通的 OpenClaw 飞书 App 通道，Baby 项目只读取 OpenClaw 现有配置和会话信息，不修改 OpenClaw 配置。

如果 OpenClaw 通道不可用，可在 `config.yaml` 里配置 webhook 作为回退：
```yaml
notification:
  feishu_enabled: true
  feishu_webhook: "https://open.feishu.cn/open-apis/bot/v2/hook/你的webhook地址"
```

### 2.4 校准安全区域
```bash
/usr/bin/python3 calibrate_region.py
```
- 标定工具复用 `camera_server.py` 的同源画面；没有运行中的摄像头服务时会自动启动临时服务
- 启动后默认不显示旧区域，直接等待新区域输入
- 左键依次点击四个角点，第四个点后自动闭合区域
- 第五次点击会清空旧四点并重新开始
- 右键或 `u` 撤销未完成的上一个点
- `r` 重置，`s` 保存，`q` 不保存退出
- 保存安全区域后会弹出英文飞书通知项选择框，可勾选 `Crying`、`Face covered`、`Left hand exposed`、`Out of safe region`；使用鼠标或数字键切换，`Enter`/`s` 保存

## 3. 启动系统

### 3.1 带预览模式（推荐用于调试）
```bash
./start.sh
# 或者
/usr/bin/python3 main.py
```

### 3.2 后台无头模式（正式运行）
```bash
./start_headless.sh
# 或者
/usr/bin/python3 main.py --no-preview
```

### 3.3 开机自启设置
编辑 `/etc/rc.local`，在 `exit 0` 前添加:
```bash
cd /home/mxin/.openclaw/workspace/baby_sleep_supervisor
sudo -u mxin ./start_headless.sh &
```

或者使用systemd服务:
创建 `/etc/systemd/system/baby-supervisor.service`:
```ini
[Unit]
Description=Baby Sleep Supervisor
After=network.target

[Service]
Type=simple
User=mxin
WorkingDirectory=/home/mxin/.openclaw/workspace/baby_sleep_supervisor
ExecStart=/usr/bin/python3 /home/mxin/.openclaw/workspace/baby_sleep_supervisor/main.py --no-preview
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

然后启用服务:
```bash
sudo systemctl daemon-reload
sudo systemctl enable baby-supervisor
sudo systemctl start baby-supervisor
```

## 4. 功能说明

### 4.1 核心检测功能
1. **Presence 确认**: 综合姿态、人脸框、Face Mesh 和分割掩码确认画面中是否确实有婴儿，兼容侧脸/背脸睡姿
2. **哭闹检测**: Face Mesh 可靠时通过面部表情识别哭闹，侧脸不可见时不会误报哭闹
3. **踢被子检测**: Presence 和 Pose 可靠时识别肢体裸露，检测是否踢掉被子
4. **口鼻遮挡检测**: Face Mesh 可靠时只分析口鼻 ROI，脸不可见不会被当成遮挡
5. **区域检测**: 优先用躯干和头部判断婴儿是否离开预设安全区域

### 4.2 告警级别
- **Notice (注意)**: 轻微异常，只记录日志不发送通知
- **Warning (警告)**: 中等异常，如踢被子、短暂离开区域，发送通知
- **Danger (危险)**: 严重异常，如口鼻遮挡、持续哭闹，立即发送通知并抓拍

### 4.3 预览模式快捷键
- `q`: 安全退出程序（先释放摄像头，再退出）
- `h`: 显示/隐藏帮助
- `d`: 显示/隐藏检测框
- `r`: 显示/隐藏安全区域
- `s`: 显示/隐藏统计信息
- `c`: 提示单独运行区域校准工具

## 5. 性能优化建议

### 5.1 散热
树莓派5运行AI推理会产生较多热量，建议使用:
- 主动散热风扇
- 金属散热外壳
- 避免封闭环境

### 5.2 性能配置
如果运行卡顿，可以在 `config.yaml` 中调整:
```yaml
inference:
  model_complexity: 0  # 降低模型复杂度，提高速度
  inference_fps: 5     # 降低推理帧率
```

### 5.3 存储
建议使用SSD存储，好处:
- 更快的读写速度
- 更高的可靠性，适合频繁写入照片
- 更大的存储空间

## 6. 常见问题

### 6.1 摄像头无法启动
- 检查摄像头排线是否插好
- 确认 `raspi-config` 中已启用摄像头
- 运行 `libcamera-hello` 测试摄像头是否正常工作
- 如果提示摄像头被占用，先退出正在运行的 Baby 程序或其他相机程序
- 预览模式下优先用 `q` 做安全退出，不要直接关闭终端窗口

### 6.2 MediaPipe 导入失败
如果看到 `ModuleNotFoundError: No module named 'mediapipe'`，通常说明推理进程没有使用正确的虚拟环境。

当前设计要求：
- 摄像头进程使用 `/usr/bin/python3`
- 推理进程使用 `/home/mxin/.openclaw/workspace/kid_supervisor_v3/venv_311/bin/python`

请确认从项目根目录执行：
```bash
./start.sh
```
而不是直接用系统 Python 运行 `inference_client.py`。

### 6.3 标定窗口打不开
如果 `calibrate_region.py` 报 OpenCV window 或 `NULL window handler` 相关错误：
- 确认是在树莓派桌面会话中运行，而不是纯 SSH headless 环境
- 确认当前用户有图形界面显示权限
- 先运行 `libcamera-hello` 确认摄像头画面正常
- 关闭其他占用摄像头的程序后再运行标定

### 6.4 预览窗口显示很多问号
OpenCV 默认文字渲染不支持中文和 emoji。当前预览 UI 已改为英文显示，如果仍看到 `?`，说明可能还有旧版本进程未退出或本地代码未更新。

处理方式：
- 按 `q` 让主进程安全重启
- 5 秒内再按一次 `q` 安全退出后重新执行 `./start.sh`

### 6.5 飞书通知没有收到
- 确认 `notification.feishu_enabled` 为 `true`
- 确认 OpenClaw 飞书通道本身可用
- 确认 OpenClaw 最近会话中存在可用的飞书接收人
- 如果 OpenClaw 通道不可用，可配置 `notification.feishu_webhook` 作为回退
- 如果文字能收到但图片收不到，检查图片文件是否存在以及飞书 App 是否具备图片上传和消息发送权限

### 6.6 检测准确率低
- 确保摄像头安装位置合适，能清晰看到婴儿面部和身体
- 侧脸/背脸睡觉时，区域检测主要依赖姿态和头/躯干关键点；哭闹和口鼻遮挡需要 Face Mesh 可用
- 光线充足但避免强光直射
- 根据实际情况调整 `config.yaml` 中的检测阈值
- 使用 `calibrate_region.py` 重新标定安全区域

### 6.7 误报太多
- 适当提高置信度阈值
- 延长持续时间阈值
- 增大 `supervision.alert_cooldown_s`
- 调整安全区域，避免区域包含无关内容
- 区域误报时优先调高 `presence_score_threshold` 或 `region_exit_confirm_ratio`
- 婴儿局部手脚伸出导致误报时，可适当降低 `region_torso_overlap_threshold`
- 在预览模式观察 Presence、Face、Region 状态后再调参

### 6.8 系统卡顿
- 降低 `inference.model_complexity`
- 降低 `inference.inference_fps`
- 关闭暂时不需要的检测功能
- 检查树莓派散热，避免高温降频
- 优先使用 SSD 保存照片和数据库

### 6.9 按 q 后程序行为说明
预览模式下 `q` 由主进程统一处理：安全停止推理和摄像头进程，销毁 OpenCV 窗口，然后退出整个程序。确保 Camera Module 3 Wide 能被正确释放，避免下次启动时摄像头占用。

## 7. 技术支持
如遇问题，请检查:
1. 系统日志
2. `data/` 目录下的事件记录和抓拍照片
3. 配置文件是否正确
