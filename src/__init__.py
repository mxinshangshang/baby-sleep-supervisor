"""
Baby Sleep Supervisor 核心模块
使用 PEP 562 __getattr__ 延迟导入，避免 camera_server（系统 Python 无 mediapipe）
导入 src 包时触发重型依赖加载导致崩溃。
"""
import sys as _sys

_LAZY_EXPORTS = {
    'AudioCryDetector': '.audio_detector',
    'fuse_audio_visual_cry': '.audio_detector',
    'get_config': '.config',
    'load_config': '.config',
    'save_config': '.config',
    'Notifier': '.notifier',
    'Storage': '.storage',
    'SleepSupervisor': '.supervision',
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_path = _LAZY_EXPORTS[name]
        module = __import__(module_path, fromlist=[name], level=1)
        attr = getattr(module, name)
        # Cache in globals so subsequent lookups skip __getattr__
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

