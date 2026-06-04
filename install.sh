#!/bin/bash
# 婴儿睡眠监护系统安装脚本
# 安装系统依赖和Python包

echo "============================================="
echo "婴儿睡眠监护系统 - 环境安装脚本"
echo "============================================="
echo ""

# 检查是否是root用户
if [ "$EUID" -eq 0 ]; then
    echo "请不要以root用户运行此脚本，使用普通用户运行即可"
    exit 1
fi

# 更新系统包
echo "1. 更新系统包列表..."
sudo apt update
echo ""

# 安装系统依赖
echo "2. 安装系统依赖..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-picamera2 \
    libopencv-dev \
    libportaudio2 \
    libasound2-dev \
    libatlas-base-dev \
    build-essential \
    cmake \
    git \
    v4l-utils
echo ""

# 检查Python版本
echo "3. 检查Python版本..."
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.9"
if [ $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc) -ne 1 ]; then
    echo "错误: Python版本需要 >= 3.9，当前版本: $PYTHON_VERSION"
    echo "请安装Python 3.9或更高版本"
    exit 1
fi
echo "Python版本: $PYTHON_VERSION ✓"
echo ""

# 创建虚拟环境
echo "4. 创建Python虚拟环境..."
cd "$(dirname "$0")"

if [ -d "venv_311" ]; then
    echo "虚拟环境已存在，是否删除重建? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "删除旧虚拟环境..."
        rm -rf venv_311
    else
        echo "使用现有虚拟环境"
    fi
fi

if [ ! -d "venv_311" ]; then
    python3 -m venv venv_311
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv_311/bin/activate
echo ""

# 升级pip
echo "5. 升级pip..."
pip install --upgrade pip setuptools wheel
echo ""

# 安装Python依赖
echo "6. 安装Python依赖包..."
pip install -r requirements.txt
echo ""

# 安装MediaPipe（针对ARM架构优化）
echo "7. 安装MediaPipe..."
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
    echo "检测到ARM64架构，安装优化版本..."
    # 尝试安装预编译版本，如果失败则从源码编译
    pip install mediapipe || {
        echo "预编译版本安装失败，尝试源码编译..."
        sudo apt install -y libopenblas-dev libgfortran5
        pip install --no-binary :all: mediapipe
    }
else
    echo "检测到x86架构，安装标准版本..."
    pip install mediapipe
fi
echo ""

# 检查安装结果
echo "8. 检查安装结果..."
echo "检查Python包:"
REQUIRED_PACKAGES=(
    "opencv-python"
    "numpy"
    "pyyaml"
    "requests"
    "mediapipe"
    "ultralytics"
    "pillow"
)

all_ok=true
for package in "${REQUIRED_PACKAGES[@]}"; do
    if pip show "$package" > /dev/null 2>&1; then
        version=$(pip show "$package" | grep Version | cut -d' ' -f2)
        echo "  ✓ $package $version"
    else
        echo "  ✗ $package 未安装"
        all_ok=false
    fi
done
echo ""

# 创建必要的目录
echo "9. 创建数据目录..."
mkdir -p data/photos
echo ""

# 设置权限
echo "10. 设置脚本执行权限..."
chmod +x *.py
chmod +x *.sh
echo ""

# 完成
echo "============================================="
echo "安装完成!"
echo "============================================="
echo ""
echo "后续步骤:"
echo "1. 配置飞书机器人 webhook:"
echo "   编辑 config.yaml 文件，修改 notification.feishu_webhook 为你的飞书机器人webhook地址"
echo ""
echo "2. 校准安全区域:"
echo "   运行 ./calibrate_region.py，用鼠标框选婴儿床区域"
echo ""
echo "3. 启动系统:"
echo "   带预览模式: ./start.sh 或 python3 main.py"
echo "   后台模式: ./start_headless.sh 或 python3 main.py --no-preview"
echo ""
echo "4. 如需要开机自启，可以将启动命令添加到 /etc/rc.local 或使用 systemd 服务"
echo ""

if [ "$all_ok" = true ]; then
    echo "所有依赖安装成功 ✓"
else
    echo "部分依赖安装失败，请手动检查 ✗"
    exit 1
fi
