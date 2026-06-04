"""
通知模块
支持飞书机器人通知，可发送文本和图片
"""
import os
import time
import json
import base64
import hashlib
import hmac
import requests
from typing import Dict, Optional
from PIL import Image
import io
from datetime import datetime

from src.config import get_config


class Notifier:
    def __init__(self):
        config = get_config()
        notify_cfg = config.get("notification", {})

        self.feishu_enabled = notify_cfg.get("feishu_enabled", True)
        self.feishu_webhook = notify_cfg.get("feishu_webhook", "")
        self.feishu_secret = notify_cfg.get("feishu_secret", "")
        self.alert_level = notify_cfg.get("alert_level", "warning")
        self.alert_cooldown = notify_cfg.get("alert_cooldown_s", 60)

        self.console_enabled = notify_cfg.get("console_enabled", True)
        self.capture_photo = notify_cfg.get("capture_photo_on_alert", True)

        # 告警级别映射
        self.level_priority = {
            "notice": 0,
            "warning": 1,
            "danger": 2
        }

        # 最后发送时间记录，用于冷却
        self.last_send_time: Dict[str, float] = {}

        # 状态模板
        self.status_emoji = {
            "notice": "ℹ️",
            "warning": "⚠️",
            "danger": "🚨"
        }

        self.status_text = {
            "notice": "注意",
            "warning": "警告",
            "danger": "危险"
        }

    def _should_send_alert(self, event_type: str, level: str) -> bool:
        """检查是否应该发送告警"""
        # 检查级别是否足够
        if self.level_priority[level] < self.level_priority[self.alert_level]:
            return False

        # 检查冷却时间
        now = time.time()
        last_time = self.last_send_time.get(event_type, 0)
        if now - last_time < self.alert_cooldown:
            return False

        return True

    def _generate_feishu_sign(self, timestamp: int) -> str:
        """生成飞书签名"""
        if not self.feishu_secret:
            return ""

        string_to_sign = f"{timestamp}\n{self.feishu_secret}"
        hmac_code = hmac.new(
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()
        sign = base64.b64encode(hmac_code).decode("utf-8")
        return sign

    def _resize_image(self, image_path: str, max_size: int = 10 * 1024 * 1024) -> bytes:
        """调整图片大小，确保不超过飞书限制"""
        with Image.open(image_path) as img:
            # 调整分辨率
            max_dimension = 1920
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # 保存到字节流
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)
            img_bytes = img_byte_arr.getvalue()

            # 如果还是太大，继续降低质量
            while len(img_bytes) > max_size and quality > 10:
                quality -= 10
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=quality)
                img_bytes = img_byte_arr.getvalue()

            return img_bytes

    def send_feishu_text(self, content: str) -> bool:
        """发送文本消息到飞书"""
        if not self.feishu_enabled or not self.feishu_webhook:
            return False

        timestamp = int(time.time())
        sign = self._generate_feishu_sign(timestamp)

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "timestamp": str(timestamp),
            "sign": sign,
            "msg_type": "text",
            "content": {
                "text": content
            }
        }

        try:
            response = requests.post(
                self.feishu_webhook,
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"飞书通知发送失败: {e}")
            return False

    def send_feishu_image(self, image_path: str, title: str = "") -> bool:
        """发送图片消息到飞书"""
        if not self.feishu_enabled or not self.feishu_webhook or not os.path.exists(image_path):
            return False

        try:
            # 上传图片
            img_bytes = self._resize_image(image_path)
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            timestamp = int(time.time())
            sign = self._generate_feishu_sign(timestamp)

            headers = {
                "Content-Type": "application/json"
            }

            payload = {
                "timestamp": str(timestamp),
                "sign": sign,
                "msg_type": "image",
                "content": {
                    "image": img_base64
                }
            }

            response = requests.post(
                self.feishu_webhook,
                headers=headers,
                data=json.dumps(payload),
                timeout=15
            )

            return response.status_code == 200
        except Exception as e:
            print(f"飞书图片发送失败: {e}")
            return False

    def send_alert(self, event_type: str, level: str, message: str,
                   photo_path: Optional[str] = None, details: Optional[Dict] = None) -> bool:
        """发送告警通知
        Args:
            event_type: 事件类型，用于冷却控制
            level: 告警级别: notice/warning/danger
            message: 告警消息
            photo_path: 抓拍图片路径（可选）
            details: 详细信息（可选）
        """
        if not self._should_send_alert(event_type, level):
            return False

        now = time.time()
        self.last_send_time[event_type] = now

        # 构造消息内容
        emoji = self.status_emoji.get(level, "ℹ️")
        status = self.status_text.get(level, "通知")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        content = f"{emoji} 婴儿监护 {status} {emoji}\n"
        content += f"时间: {current_time}\n"
        content += f"事件: {message}\n"

        if details:
            content += "\n详细信息:\n"
            for key, value in details.items():
                if isinstance(value, float):
                    content += f"  {key}: {value:.2f}\n"
                else:
                    content += f"  {key}: {value}\n"

        # 控制台输出
        if self.console_enabled:
            print(f"\n{content}")
            if photo_path:
                print(f"  抓拍图片: {photo_path}")

        # 飞书通知
        if self.feishu_enabled:
            # 先发送文本
            self.send_feishu_text(content)

            # 再发送图片（如果有）
            if photo_path and self.capture_photo and os.path.exists(photo_path):
                time.sleep(0.5)  # 避免发送太快
                self.send_feishu_image(photo_path, title=message)

        return True

    def send_system_notification(self, message: str) -> bool:
        """发送系统通知"""
        content = f"🔔 系统通知\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{message}"

        if self.console_enabled:
            print(f"\n{content}")

        if self.feishu_enabled:
            return self.send_feishu_text(content)

        return True
