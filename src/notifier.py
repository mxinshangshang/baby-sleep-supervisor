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
import re
import requests
from typing import Dict, Optional, Tuple
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
        self.enabled_alert_types = set(notify_cfg.get("enabled_alert_types", [
            "cry_detected",
            "occlusion_detected",
            "limb_exposure",
            "region_exit",
            "prone_detected",
        ]))

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

        self.openclaw_cfg, self.openclaw_receive_id = self._load_openclaw_feishu_config()

    def _should_send_alert(self, event_type: str, level: str) -> bool:
        """检查是否应该发送告警"""
        if event_type not in self.enabled_alert_types:
            return False

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

    def _load_openclaw_feishu_config(self) -> Tuple[Optional[Dict], Optional[str]]:
        """复用OpenClaw已打通的飞书App配置和最近会话ID"""
        config_path = "/home/mxin/.openclaw/openclaw.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                oc_cfg = json.load(f)
            feishu_cfg = oc_cfg.get("channels", {}).get("feishu", {})
            if not feishu_cfg.get("enabled"):
                return None, None

            receive_id = self._find_openclaw_feishu_receive_id()
            if not receive_id:
                print("OpenClaw Feishu receive_id not found; text webhook fallback will be used")
                return feishu_cfg, None
            return feishu_cfg, receive_id
        except Exception as e:
            print(f"OpenClaw Feishu config load failed: {e}")
            return None, None

    def _find_openclaw_feishu_receive_id(self) -> Optional[str]:
        """从OpenClaw会话索引中提取最近的飞书open_id"""
        candidates = [
            "/home/mxin/.openclaw/agents/claude/sessions/sessions.json",
            "/home/mxin/.openclaw/agents/main/sessions/sessions.json",
        ]
        pattern = re.compile(r"feishu:(?:direct|group):([A-Za-z0-9_\-]+)")
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                matches = pattern.findall(text)
                if matches:
                    return matches[-1]
            except Exception:
                continue
        return None

    def _openclaw_tenant_access_token(self) -> Optional[str]:
        if not self.openclaw_cfg:
            return None
        try:
            response = requests.post(
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
                json={
                    "app_id": self.openclaw_cfg.get("appId"),
                    "app_secret": self.openclaw_cfg.get("appSecret"),
                },
                timeout=10,
            )
            data = response.json()
            if data.get("code") == 0:
                return data.get("tenant_access_token")
            print(f"OpenClaw Feishu token failed: {data}")
        except Exception as e:
            print(f"OpenClaw Feishu token request failed: {e}")
        return None

    def _openclaw_send_text(self, content: str) -> bool:
        if not self.openclaw_receive_id:
            return False
        token = self._openclaw_tenant_access_token()
        if not token:
            return False
        try:
            response = requests.post(
                "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "receive_id": self.openclaw_receive_id,
                    "msg_type": "text",
                    "content": json.dumps({"text": content}, ensure_ascii=False),
                },
                timeout=10,
            )
            data = response.json()
            if data.get("code") == 0:
                return True
            print(f"OpenClaw Feishu text send failed: {data}")
        except Exception as e:
            print(f"OpenClaw Feishu text send failed: {e}")
        return False

    def _openclaw_upload_image(self, image_path: str) -> Optional[str]:
        token = self._openclaw_tenant_access_token()
        if not token:
            return None
        try:
            img_bytes = self._resize_image(image_path)
            files = {"image": (os.path.basename(image_path), img_bytes, "image/jpeg")}
            data = {"image_type": "message"}
            response = requests.post(
                "https://open.feishu.cn/open-apis/im/v1/images",
                headers={"Authorization": f"Bearer {token}"},
                data=data,
                files=files,
                timeout=20,
            )
            payload = response.json()
            if payload.get("code") == 0:
                return payload.get("data", {}).get("image_key")
            print(f"OpenClaw Feishu image upload failed: {payload}")
        except Exception as e:
            print(f"OpenClaw Feishu image upload failed: {e}")
        return None

    def _openclaw_send_image(self, image_path: str) -> bool:
        if not self.openclaw_receive_id or not os.path.exists(image_path):
            return False
        token = self._openclaw_tenant_access_token()
        if not token:
            return False
        image_key = self._openclaw_upload_image(image_path)
        if not image_key:
            return False
        try:
            response = requests.post(
                "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "receive_id": self.openclaw_receive_id,
                    "msg_type": "image",
                    "content": json.dumps({"image_key": image_key}),
                },
                timeout=10,
            )
            data = response.json()
            if data.get("code") == 0:
                return True
            print(f"OpenClaw Feishu image send failed: {data}")
        except Exception as e:
            print(f"OpenClaw Feishu image send failed: {e}")
        return False

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
            quality = 85
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=quality)
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
        if self.feishu_enabled and self._openclaw_send_text(content):
            return True

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
        if self.feishu_enabled and self._openclaw_send_image(image_path):
            return True

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
