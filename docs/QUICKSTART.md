# 快速上手指南

---

## 第1步：确认环境

本项目已预配置好，不需要重复安装依赖。

**验证环境**:
```bash
# 确认摄像头存在
libcamera-hello --list-cameras

# 确认Python环境
/usr/bin/python3 --version                  # 系统Python - 摄像头用
/home/mxin/.openclaw/workspace/kid_supervisor_v3/venv_311/bin/python --version  # 推理用
```

✅ 两个Python都能正常运行 → 继续

---

## 第2步：区域校准 (最重要)

```bash
cd /home/mxin/.openclaw/workspace/baby_sleep_supervisor
/usr/bin/python3 calibrate_region.py
```

**校准操作**:

1. 窗口弹出后，你会看到实时画面
2. **依次左键点击婴儿床四个角**：
   - 点1：左上角
   - 点2：右上角
   - 点3：右下角
   - 点4：左下角
3. 点完第四个点，绿色多边形自动闭合
4. 不满意？点击**第五个点** → 自动清除，重新画
5. 按 `u` 撤销上一个点
6. 按 `r` 全部重置
7. 位置满意后 → **按 `s` 保存**

8. 最后一步：选择需要飞书通知的告警类型
   - 鼠标点击或按数字键 1-7 勾选/取消
   - `a` 全选，`n` 全不选
   - **Enter** 或 **s** 确认保存

✅ 区域配置完成！

---

## 第3步：配置飞书通知 (可选但推荐)

编辑 `config.yaml`:

```yaml
notification:
  feishu_enabled: true
  feishu_webhook: "https://open.feishu.cn/open-apis/bot/v2/hook/你的token"

  # 勾选你需要的通知类型
  enabled_alert_types:
    - cry_detected        # 哭闹
    - occlusion_detected  # 口鼻遮挡
    - limb_exposure       # 踢被子/肢体裸露
    - region_exit         # 离开安全区域
    - region_enter        # 进入安全区域
    - prone_detected      # 趴睡风险
    - face_not_visible    # 面部不可见
```

---

## 第4步：启动系统

### 方式A：桌面调试 - 带预览

```bash
./start.sh
```

你会看到：
- 实时视频画面（暗光自动切换夜视摄像头）
- 彩色检测框
- 左上角状态数值
- 右下角快捷键帮助

按 `q` 安全退出。

### 方式B：生产部署 - 无头模式

```bash
/usr/bin/python3 main.py -n
```

纯后台运行，不显示窗口，资源占用更低。

**停止服务**:
```bash
pkill -f "python3.*main.py"
```

---

## 第5步：验证运行

### 检查进程

```bash
ps aux | grep python
```

应该看到 **3个进程**:
1. `python3 main.py` - 主启动器
2. `/usr/bin/python3 camera_server.py` - 摄像头进程
3. `...venv_311/bin/python inference_client.py` - 推理进程

### 检查日志

```bash
# 摄像头服务器日志（含双摄切换记录）
tail -f /tmp/camera_server.log

# 主进程 + 推理客户端日志
tail -f /tmp/baby_preview.log
```

正常输出示例：
```
[Camera] CameraRouter初始化完成（双摄模式）
[Camera] 服务器启动，等待客户端连接: 127.0.0.1:65433

运行中... preview_fps=10.0 infer_fps=3.0 camera_seq=1234 temp=68.8C
```

---

## 常见问题速查

### Q: 启动后画面全黑？

**A**: 暗光环境下常规摄像头拍出来是黑的，系统会自动切换夜视摄像头。首次切换约需 14 秒（`stable_frames=7` × 每 30 帧检测一次）。查看 `/tmp/camera_server.log` 确认切换日志：
```
亮度=0.002 active=0 dark=7/7 → 切换完成 0 -> 1
```

### Q: 双摄没有生效？

**A**: 确认 `config.yaml` 中 `dual_camera.enabled: true`。检查 `/tmp/camera_server.log` 是否有 `CameraRouter初始化完成（双摄模式）`。

### Q: 检测框一直不出现？

**A**: 确认宝宝在画面中央，调整角度让脸部和身体都可见。MediaPipe 需要一定的清晰度和角度才能检测到关键点。

### Q: 误报太多？哭闹太灵敏？遮挡误报？

**A**: 调整阈值（`config.yaml`）:

```yaml
detection:
  cry_confidence_threshold: 0.6    # 从0.5调到0.6或0.7
  cry_duration_threshold: 3.0      # 从2秒调到3秒
  occlusion_threshold: 0.85        # 从0.75调高

notification:
  alert_cooldown_s: 30.0           # 冷却时间加长，减少重复通知
```

### Q: 漏报？哭闹没检测到？

**A**: 调低阈值增加灵敏度:

```yaml
detection:
  cry_confidence_threshold: 0.4    # 更灵敏
  cry_duration_threshold: 1.5      # 更快响应
```

⚠️ 注意：灵敏度太高会增加误报，建议逐步微调观察24小时。

### Q: CPU温度很高？

**A**: 正常现象，树莓派5满负载会到70-80°C。系统有自动温控:

```yaml
thermal:
  temp_warn_c: 65    # ≥65°C 自动降推理帧率
  temp_throttle_c: 75 # ≥75°C 自动切换轻量模型
```

如果持续过热，可以主动降低推理帧率:
```yaml
inference:
  inference_fps: 2  # 从3降到2，CPU降约30%
```

### Q: 怎么看历史事件和回溯分析？

```bash
# 查看告警事件
sqlite3 data/events.db "SELECT * FROM events ORDER BY timestamp DESC LIMIT 10;"

# 查看诊断快照（用于回溯分析历史误报/漏报）
sqlite3 data/events.db "SELECT timestamp, json_extract(snapshot_json, '$.alerts') FROM events_debug WHERE timestamp BETWEEN '2026-06-19 17:12:00' AND '2026-06-19 17:14:00';"

# 查看抓拍照片
ls -la data/photos/
```

### Q: 如何手动切换摄像头？

```bash
# 发送 SIGUSR2 给 camera_server 强制切换
kill -USR2 $(pgrep -f camera_server.py)
```

---

## 日常运维

### 查看系统状态

```bash
# CPU温度
vcgencmd measure_temp

# 磁盘使用
du -sh data/photos/

# 运行时间
ps -eo pid,etime,cmd | grep python | grep -v grep
```

### 重启服务

```bash
pkill -f "python3.*main.py"
sleep 2
./start.sh
```

### 清理旧数据

```bash
# 系统会自动清理30天前的数据，也可以手动清理
find data/photos/ -name "*.jpg" -mtime +30 -delete
```

---

## 性能参考 (树莓派5 8GB)

| 模式 | CPU占用 | 内存占用 | 温度 |
|------|---------|----------|------|
| 预览模式 | ~180% (4核) | ~350MB | 65-72°C |
| 无头模式 | ~150% | ~300MB | 62-68°C |
| 75°C降频后 | ~120% | ~280MB | 68-73°C |

✅ **可持续24/7稳定运行**

---

## 下一步

- 阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 深入了解系统设计
- 根据实际使用情况微调 `config.yaml` 参数
- 观察24小时运行效果，逐步优化灵敏度

**有问题？先看日志输出，大部分问题都有明确的错误提示！**
