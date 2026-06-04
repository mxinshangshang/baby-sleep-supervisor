"""
配置加载模块
"""
import os
import yaml
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# 全局配置缓存
_global_config: Dict[str, Any] | None = None


def load_config(base_dir: str = None) -> Dict[str, Any]:
    """加载配置文件"""
    global _global_config

    if base_dir is None:
        base_dir = BASE_DIR

    config_path = os.path.join(base_dir, "config.yaml")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 确保目录存在
        storage_cfg = config.get("storage", {})
        os.makedirs(os.path.join(base_dir, storage_cfg.get("photo_dir", "data/photos")), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(base_dir, storage_cfg.get("sqlite_path", "data/events.db"))), exist_ok=True)

        _global_config = config
        return config
    except Exception as e:
        raise RuntimeError(f"配置加载失败: {e}") from e


def get_config() -> Dict[str, Any]:
    """获取全局配置"""
    if _global_config is None:
        return load_config()
    return _global_config


def save_config(config: Dict[str, Any]) -> None:
    """保存配置到文件"""
    global _global_config
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        _global_config = config
    except Exception as e:
        raise RuntimeError(f"配置保存失败: {e}") from e
