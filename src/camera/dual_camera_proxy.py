"""
CameraRouter（双摄像头互斥路由器）
设计原则：零侵入，对外暴露与Picamera2同名的鸭子接口！
"""
from __future__ import annotations
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional
import numpy as np
from picamera2 import Picamera2
from src.camera.ambient_light_detector import AmbientLightDetector

logger = logging.getLogger("CameraRouter")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False
_DEFAULT_WARMUP_S = 2.0

class CameraRouter:
    def __init__(self, config_builder: Callable[[Picamera2], Any], dual_cfg: Dict[str, Any], storage: Optional[Any] = None):
        self._config_builder = config_builder
        self._storage = storage
        self.normal_id = int(dual_cfg.get("normal_camera_id", 0))
        self.night_id = int(dual_cfg.get("night_camera_id", 1))
        self.default_id = int(dual_cfg.get("default_camera_id", self.normal_id))
        self.check_interval = int(dual_cfg.get("check_interval_frames", 30))
        self.warmup_overlap_s = float(dual_cfg.get("warmup_overlap_s", _DEFAULT_WARMUP_S))
        self._light = AmbientLightDetector(dual_cfg.get("light_detection", {}))
        self._active = None
        self._active_id = self.default_id
        self._last_frame = None
        self._frame_count = 0
        self._last_switch_time = 0
        self._switching = False
        self._switch_lock = threading.Lock()
        self._stopped = False

    def configure(self, _unused: Any = None):
        pass

    def start(self):
        if self._active:
            logger.warning("已经启动，忽略")
            return
        self._active = self._build_and_start(self.default_id)
        self._active_id = self.default_id
        logger.info(f"CameraRouter已启动，active={self._active_id}")

    def stop(self):
        self._stopped = True
        cam = self._active
        self._active = None
        if cam:
            self._safe_close(cam, self._active_id)

    def capture_array(self):
        cam = self._active
        if not cam:
            if self._last_frame is not None:
                return self._last_frame
            raise RuntimeError("未启动或已停止")
        try:
            frame = cam.capture_array()
        except Exception as e:
            logger.warning(f"capture_array失败: {e}，返回最后一帧")
            if self._last_frame is not None:
                return self._last_frame
            raise
        self._last_frame = frame
        self._frame_count += 1
        if self.check_interval > 0 and not self._switching and self._frame_count % self.check_interval == 0:
            self._maybe_trigger_switch(frame)
        return frame

    def camera_configuration(self):
        if not self._active:
            return {}
        try:
            return self._active.camera_configuration()
        except Exception:
            return {}

    def _build_and_start(self, camera_id: int):
        cam = Picamera2(camera_id)
        config = self._config_builder(cam)
        cam.configure(config)
        cam.start()
        time.sleep(self.warmup_overlap_s)
        return cam

    def _safe_close(self, cam, cam_id):
        try:
            cam.stop()
        except Exception as e:
            logger.warning(f"停止cam{cam_id}失败: {e}")
        try:
            cam.close()
        except Exception as e:
            logger.warning(f"关闭cam{cam_id}失败: {e}")

    def _maybe_trigger_switch(self, frame):
        time_since = time.time() - self._last_switch_time
        target = self._light.should_switch(frame, current_camera=self._active_id, time_since_last_switch=time_since)
        if self._frame_count % (self.check_interval * 3) == 0:
            b = self._light.brightness_window[-1] if self._light.brightness_window else 0
            logger.info(f"亮度={b:.3f} active={self._active_id} dark={self._light.consecutive_dark}/{self._light.stable_frames} bright={self._light.consecutive_bright}/{self._light.stable_frames} time_since_switch={time_since:.0f}s")
        if target is None or target == self._active_id or target not in (self.normal_id, self.night_id):
            return
        threading.Thread(target=lambda: self._do_switch(target, "ambient_light"), daemon=True).start()

    def force_switch(self, target_id: int, reason: str = "manual"):
        if target_id == self._active_id:
            return True
        if target_id not in (self.normal_id, self.night_id):
            logger.error(f"目标cam{target_id}不在配置里")
            return False
        return self._do_switch(target_id, reason)

    def _do_switch(self, target_id, reason):
        if not self._switch_lock.acquire(blocking=False):
            logger.info("已有切换在进行，忽略")
            return False
        if self._stopped:
            self._switch_lock.release()
            return False
        old_id = self._active_id
        old_cam = self._active
        switch_start = time.time()
        self._switching = True
        try:
            logger.info(f"切换开始 {old_id} -> {target_id} (reason={reason})")
            new_cam = self._build_and_start(target_id)
            self._active = new_cam
            self._active_id = target_id
            self._last_switch_time = time.time()
            self._light.reset()
            if old_cam:
                threading.Thread(target=lambda: self._safe_close(old_cam, old_id), daemon=True).start()
            elapsed_ms = int((time.time() - switch_start) * 1000)
            logger.info(f"切换完成 {old_id} -> {target_id}，耗时 {elapsed_ms}ms")
            if self._storage:
                try:
                    self._storage.save_event(event_type="camera_switch", level="info", message=f"camera switch {old_id}->{target_id}", payload={"from": old_id, "to": target_id, "reason": reason, "elapsed_ms": elapsed_ms}, update_stats=False)
                except Exception:
                    pass
            return True
        except Exception as e:
            logger.error(f"切换失败，保持cam{self._active_id}: {e}")
            if self._storage:
                try:
                    self._storage.save_event(event_type="camera_switch_error", level="warning", message=f"camera switch failed", payload={"from": old_id, "to": target_id, "reason": reason, "error": str(e)}, update_stats=False)
                except Exception:
                    pass
            return False
        finally:
            self._switching = False
            self._switch_lock.release()

    def get_status(self):
        return {"dual_enabled": True, "active_camera_id": self._active_id, "normal_camera_id": self.normal_id, "night_camera_id": self.night_id, "is_switching": self._switching, "frame_count": self._frame_count, "last_switch_time": self._last_switch_time}
