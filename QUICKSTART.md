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

### 2.2 安装依赖
```bash
# 进入项目目录
cd /home/mxin/.openclaw/workspace/baby_sleep_supervisor

# 运行完整安装脚本
./install.sh

# 或者快速安装（基础功能）
# ./install_fast.sh
```

### 2.3 配置飞书通知（可选）
1. 在飞书创建自定义机器人，获取webhook地址
2. 编辑 `config.yaml` 文件:
```yaml
notification:
  feishu_webhook: "https://open.feishu.cn/open-apis/bot/v2/hook/你的webhook地址"
```

### 2.4 校准安全区域
```bash
./calibrate_region.py
```
- 用鼠标在画面上拖动框选婴儿床区域
- 按 `s` 保存配置
- 按 `q` 退出

## 3. 启动系统

### 3.1 带预览模式（推荐用于调试）
```bash
./start.sh
# 或者
python3 main.py
```

### 3.2 后台无头模式（正式运行）
```bash
./start_headless.sh
# 或者
python3 main.py --no-preview
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
ExecStart=/usr/bin/python3 main.py --no-preview
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
1. **哭闹检测**: 通过面部表情识别婴儿哭闹，支持不同灵敏度设置
2. **踢被子检测**: 识别肢体裸露，检测是否踢掉被子
3. **口鼻遮挡检测**: 检测被褥、玩具等遮挡口鼻，防止窒息风险
4. **区域检测**: 检测婴儿是否离开预设的安全区域

### 4.2 告警级别
- **Notice (注意)**: 轻微异常，只记录日志不发送通知
- **Warning (警告)**: 中等异常，如踢被子、短暂离开区域，发送通知
- **Danger (危险)**: 严重异常，如口鼻遮挡、持续哭闹，立即发送通知并抓拍

### 4.3 预览模式快捷键
- `q`: 退出程序
- `h`: 显示/隐藏帮助
- `d`: 显示/隐藏检测框
- `r`: 显示/隐藏安全区域
- `s`: 显示/隐藏统计信息
- `c`: 校准安全区域（需单独运行校准工具）

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
- 确认raspi-config中已启用摄像头
- 运行 `libcamera-hello` 测试摄像头是否正常工作

### 6.2 检测准确率低
- 确保摄像头安装位置合适，能清晰看到婴儿面部和身体
- 光线充足但避免强光直射
- 根据实际情况调整 `config.yaml` 中的检测阈值

### 6.3 误报太多
- 适当提高置信度阈值
- 延长持续时间阈值
- 调整安全区域，避免区域包含无关内容

### 6.4 系统卡顿
- 降低模型复杂度
- 降低推理帧率
- 关闭不需要的检测功能（如不需要声音检测可以关闭）

## 7. 技术支持
如遇问题，请检查:
1. 系统日志
2. `data/` 目录下的事件记录和抓拍照片
3. 配置文件是否正确
