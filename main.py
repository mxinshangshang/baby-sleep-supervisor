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
import traceback
from dataclasses import dataclass

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def log_crash(process_name: str, reason: str, exit_code: int = None, exc_info=None):
    """记录崩溃/退出原因到日志文件"""
    try:
        log_path = os.path.join(BASE_DIR, "data", "crash_log.txt")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"[{timestamp}] 主进程监控: {process_name} - {reason}\n")
            if exit_code is not None:
                f.write(f"退出码: {exit_code}\n")
            if exc_info and exc_info[0] is not None:
                f.write(f"异常类型: {exc_info[0].__name__}\n")
                f.write(f"异常信息: {exc_info[1]}\n")
                f.write("堆栈追踪:\n")
                traceback.print_tb(exc_info[2], file=f)
            f.write(f"{'='*60}\n")
    except Exception:
        pass

# Python 路径（和kid_supervisor完全一致：摄像头用系统Python，推理用3.11虚拟环境）
SYSTEM_PYTHON = "/usr/bin/python3"
# 直接复用kid项目已经配好的虚拟环境，零配置直接用，不需要额外安装任何东西
VENV_PYTHON = "/home/mxin/.openclaw/workspace/kid_supervisor_v3/venv_311/bin/python"

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
    """和kid_supervisor一致，不需要虚拟环境，直接返回True"""
    return True


def main():
    no_preview = "--no-preview" in sys.argv or "-n" in sys.argv

    print("=" * 60)
    print("Baby Sleep Supervisor v1.0 - 双进程架构 (自动重启)")
    print(f"预览: {'禁用 (headless)' if no_preview else '启用'}")
    print("提示: 按q安全退出，确保摄像头资源释放")
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
    shutting_down = False  # 主动关闭标志：True时子进程退出不再重启
    last_status_log = 0.0

    def cleanup(signum=None, frame=None):
        nonlocal running, shutting_down
        running = False
        shutting_down = True  # 设置主动关闭标志，防止看门狗误重启
        print("\n[Main] 正在安全退出，释放所有资源...")
        # 先终止推理进程
        if inference_state.proc and inference_state.proc.poll() is None:
            print(f"[Main] 终止推理进程 (PID {inference_state.proc.pid})")
            inference_state.proc.terminate()
            try:
                inference_state.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                inference_state.proc.kill()
                inference_state.proc.wait()
        # 再终止摄像头进程，确保摄像头设备先释放
        if camera_state.proc and camera_state.proc.poll() is None:
            print(f"[Main] 终止摄像头进程 (PID {camera_state.proc.pid})")
            camera_state.proc.terminate()
            try:
                camera_state.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                camera_state.proc.kill()
                camera_state.proc.wait()
        # 清理可能残留的opencv窗口
        try:
            import cv2
            cv2.destroyAllWindows()
        except:
            pass
        print("[Main] 所有资源已释放，安全退出")

    def handle_q_press(signum, frame):
        """处理来自子进程的q按键信号"""
        print("\n[Main] 检测到q按键，执行安全退出...")
        cleanup()

    signal.signal(signal.SIGINT, lambda s, f: cleanup(s, f))
    signal.signal(signal.SIGTERM, lambda s, f: cleanup(s, f))
    signal.signal(signal.SIGUSR1, handle_q_press)  # 接收子进程的q按键信号

    def start_camera():
        print("[Main] 启动摄像头服务器 (系统 Python)...")
        camera_state.proc = subprocess.Popen([SYSTEM_PYTHON, CAMERA_SCRIPT])
        camera_state.started_at = time.time()
        print(f"[Main] 摄像头服务器 PID: {camera_state.proc.pid}")

    def start_inference():
        print("[Main] 启动推理客户端 (复用kid项目虚拟环境)...")
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

        # 记录退出原因
        exit_reason = "主动关闭" if shutting_down else "异常崩溃"
        log_crash(state.name, exit_reason, state.last_exit_code)
        print(f"[Main] {state.name} 退出 (code: {state.last_exit_code})")

        # 主动关闭时不再重启
        if shutting_down:
            return None

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

            # 故障自动重启逻辑
            if camera_state.proc and camera_state.proc.poll() is not None:
                rc = restart_or_exit(camera_state, start_camera)
                if rc is not None:
                    return rc

            if inference_state.proc and inference_state.proc.poll() is not None:
                rc = restart_or_exit(inference_state, start_inference)
                if rc is not None:
                    return rc

    except KeyboardInterrupt:
        log_crash("主进程", "用户主动 Ctrl+C 停止")
        raise
    except SystemExit as e:
        log_crash("主进程", f"系统主动退出 (code={e.code})")
        raise
    except BaseException as e:
        log_crash("主进程", "异常崩溃", exc_info=sys.exc_info())
        raise
    finally:
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
