#!/usr/bin/env python3
"""
Baby Sleep Supervisor 虚拟环境搭建脚本
自动创建 Python 3.11 虚拟环境并安装依赖
"""
import os
import sys
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, "venv")
PYTHON_BIN = sys.executable  # 用当前运行的Python
REQUIREMENTS = os.path.join(BASE_DIR, "requirements.txt")


def check_python_version():
    """检查 Python 版本是否符合要求 (>=3.9)"""
    result = subprocess.run([PYTHON_BIN, "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Python 执行失败: {result.stderr}")
        return False

    version_info = sys.version_info
    if version_info.major < 3 or version_info.minor < 9:
        print("=" * 60)
        print(f"错误：Python 版本需要 >= 3.9，当前版本: {sys.version}")
        print("=" * 60)
        return False

    print(f"Python 版本: {result.stdout.strip()} ✓")
    return True


def create_venv():
    """创建虚拟环境"""
    if os.path.exists(VENV_DIR):
        print(f"\n虚拟环境已存在，是否删除重建? (y/N)")
        choice = input().strip().lower()
        if choice == 'y':
            print("正在删除旧虚拟环境...")
            shutil.rmtree(VENV_DIR)
        else:
            print("使用现有虚拟环境")
            return True

    print("\n正在创建 Python 虚拟环境...")
    result = subprocess.run([PYTHON_BIN, "-m", "venv", VENV_DIR])
    if result.returncode != 0:
        print("虚拟环境创建失败")
        return False

    print("虚拟环境创建成功")
    return True


def install_dependencies():
    """安装依赖包"""
    pip_path = os.path.join(VENV_DIR, "bin", "pip")
    python_path = os.path.join(VENV_DIR, "bin", "python")

    print("\n正在升级 pip...")
    result = subprocess.run([pip_path, "install", "--upgrade", "pip", "setuptools", "wheel"])
    if result.returncode != 0:
        print("pip 升级失败")
        return False

    print("\n正在安装依赖包...")
    result = subprocess.run([pip_path, "install", "-r", REQUIREMENTS])
    if result.returncode != 0:
        print("依赖安装失败")
        return False

    # 安装系统级依赖提示
    print("\n" + "=" * 60)
    print("系统依赖安装提示")
    print("=" * 60)
    print("\n请确保已安装以下系统依赖:")
    print("sudo apt install -y libopencv-dev libportaudio2 libasound2-dev")
    print("sudo apt install -y python3-picamera2 libcamera-apps")
    print("\n对于摄像头支持，请确保已启用摄像头接口:")
    print("sudo raspi-config -> Interface Options -> Camera -> Enable\n")

    return True


def main():
    print("=" * 60)
    print("Baby Sleep Supervisor 环境搭建工具")
    print("=" * 60)

    if not check_python_version():
        return 1

    if not create_venv():
        return 1

    if not install_dependencies():
        return 1

    print("=" * 60)
    print("环境搭建完成!")
    print("=" * 60)
    print(f"\n虚拟环境路径: {VENV_DIR}")
    print("\n启动命令:")
    print("  带预览模式: ./start.sh")
    print("  无头模式:   ./start_headless.sh")
    print("\n区域校准:")
    print("  ./calibrate_region.py")
    print("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
