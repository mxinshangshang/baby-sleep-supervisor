#!/bin/bash
# 快速安装脚本
# 只安装必要的依赖，跳过可选组件

echo "============================================="
echo "婴儿睡眠监护系统 - 快速安装脚本"
echo "============================================="
echo ""

cd "$(dirname "$0")"

# 安装系统依赖
echo "1. 安装必要系统依赖..."
sudo apt install -y python3-pip python3-venv python3-picamera2 libopencv-dev
echo ""

# 创建虚拟环境
echo "2. 创建虚拟环境..."
if [ ! -d "venv_311" ]; then
    python3 -m venv venv_311
fi
source venv_311/bin/activate
echo ""

# 安装基础依赖
echo "3. 安装Python依赖..."
pip install --upgrade pip
pip install opencv-python numpy pyyaml requests pillow mediapipe
echo ""

# 创建目录
mkdir -p data/photos
chmod +x *.py *.sh

echo "============================================="
echo "快速安装完成!"
echo "============================================="
echo ""
echo "注意: 快速安装只包含基础功能，如需要高级功能请运行 install.sh"
echo ""
echo "下一步: "
echo "1. 编辑 config.yaml 配置飞书webhook"
echo "2. 运行 ./calibrate_region.py 校准安全区域"
echo "3. 运行 ./start.sh 启动系统"
