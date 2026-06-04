#!/usr/bin/env python3
"""
Baby Sleep Supervisor v1.0 - 双进程架构主启动器
启动摄像头服务器和推理客户端两个进程
支持自动重启、配置化、退避策略
"""
import sys
import os
import subprocess
import signal
import time
from dataclasses import dataclass

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Python 路径
SYSTEM_PYTHON = "/usr/bin/python3"
VENV_PYTHON = os.path.join(BASE_DIR, "venv", "bin", "python")

# 脚本路径
CAMERA_SCRIPT = os.path.join(BASE_DIR, "camera_server.py")
INFERENCE_SCRIPT = os.path.join(BASE_DIR, "inference_client.py")

# 默认配置
DEFAULT_CONFIG = {
    "process": {
        "max_restart_attempts": 5,
        "restart_backoff_base_s": 2,
        "restart_reset_after_s": 60,
        "status_log_interval_s": 30
    }
}


@dataclass
class ProcessState:
    name: str
    proc: subprocess.Popen | None = None
    restart_count: int = 0
    started_at: float = 0.0
    last_exit_code: int | None = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def uptime(self, now: float) -> float:
        return max(0.0, now - self.started_at) if self.started_at else 0.0


def load_config():
    """加载配置"""
    import yaml
    config_path = os.path.join(BASE_DIR, "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"配置加载失败，使用默认配置: {e}")
        return DEFAULT_CONFIG


def check_venv():
    """检查 venv 是否存在"""
    if not os.path.exists(VENV_PYTHON):
        print("=" * 60)
        print("Python 虚拟环境不存在")
        print("=" * 60)
        print("\n请先运行: python3 setup_venv.py\n")
        return False
    return True


def main():
    no_preview = "--no-preview" in sys.argv or "-n" in sys.argv

    print("=" * 60)
    print("Baby Sleep Supervisor v1.0 - 双进程架构 (自动重启)")
    print(f"预览: {'禁用 (headless)' if no_preview else '启用'}")
    print("=" * 60)

    # 加载配置
    config = load_config()
    proc_cfg = config.get("process", DEFAULT_CONFIG["process"])
    MAX_RESTART = proc_cfg.get("max_restart_attempts", 5)
    RESTART_BACKOFF_BASE = proc_cfg.get("restart_backoff_base_s", 2)
    RESTART_RESET_AFTER = proc_cfg.get("restart_reset_after_s", 60)
    STATUS_LOG_INTERVAL = proc_cfg.get("status_log_interval_s", 30)

    if not check_venv():
        return 1

    camera_state = ProcessState(name="camera")
    inference_state = ProcessState(name="inference")
    running = True
    last_status_log = 0.0

    def cleanup(signum=None, frame=None):
        nonlocal running
        running = False
        print("\n[Main] 正在关闭子进程...")
        for name, proc in [("camera", camera_state.proc), ("inference", inference_state.proc)]:
            if proc and proc.poll() is None:
                print(f"[Main] 终止 {name} (PID {proc.pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        print("[Main] 所有进程已退出")

    signal.signal(signal.SIGINT, lambda s, f: cleanup(s, f))
    signal.signal(signal.SIGTERM, lambda s, f: cleanup(s, f))

    def start_camera():
        print("[Main] 启动摄像头服务器 (系统 Python)...")
        camera_state.proc = subprocess.Popen([SYSTEM_PYTHON, CAMERA_SCRIPT])
        camera_state.started_at = time.time()
        print(f"[Main] 摄像头服务器 PID: {camera_state.proc.pid}")

    def start_inference():
        print("[Main] 启动推理客户端 (虚拟环境 Python)...")
        args = [VENV_PYTHON, INFERENCE_SCRIPT]
        if no_preview:
            args.append("--no-preview")
        inference_state.proc = subprocess.Popen(args)
        inference_state.started_at = time.time()
        print(f"[Main] 推理客户端 PID: {inference_state.proc.pid}")

    def maybe_reset_restart_counter(state: ProcessState, now: float):
        if state.restart_count > 0 and state.is_running() and state.uptime(now) >= RESTART_RESET_AFTER:
            print(f"[Main] {state.name} 已稳定运行 {int(state.uptime(now))}s，重启计数清零")
            state.restart_count = 0

    def restart_or_exit(state: ProcessState, starter):
        state.last_exit_code = state.proc.returncode if state.proc else None
        state.restart_count += 1
        print(f"[Main] {state.name} 退出 (code: {state.last_exit_code})")

        if state.restart_count <= MAX_RESTART:
            backoff = min(RESTART_BACKOFF_BASE * state.restart_count, 10)
            print(f"[Main] {backoff}s 后重启 {state.name} ({state.restart_count}/{MAX_RESTART})")
            time.sleep(backoff)
            starter()
            return None

        print(f"[Main] {state.name} 重启次数超限 ({MAX_RESTART})，退出")
        cleanup()
        return 1

    def log_status(now: float):
        cam_status = f"up {int(camera_state.uptime(now))}s" if camera_state.is_running() else f"down rc={camera_state.last_exit_code}"
        inf_status = f"up {int(inference_state.uptime(now))}s" if inference_state.is_running() else f"down rc={inference_state.last_exit_code}"
        print(
            f"[Main Status] camera={cam_status} restarts={camera_state.restart_count} | "
            f"inference={inf_status} restarts={inference_state.restart_count}"
        )

    # 初始启动
    start_camera()
    time.sleep(2)  # 等待摄像头初始化
    start_inference()

    print("\n[Main] 两个进程都已启动，按 Ctrl+C 退出\n")

    try:
        while running:
            time.sleep(0.5)
            now = time.time()

            maybe_reset_restart_counter(camera_state, now)
            maybe_reset_restart_counter(inference_state, now)

            if now - last_status_log >= STATUS_LOG_INTERVAL:
                log_status(now)
                last_status_log = now

            if camera_state.proc and camera_state.proc.poll() is not None:
                rc = restart_or_exit(camera_state, start_camera)
                if rc is not None:
                    return rc

            if inference_state.proc and inference_state.proc.poll() is not None:
                rc = restart_or_exit(inference_state, start_inference)
                if rc is not None:
                    return rc

    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
