#!/bin/bash
# 带预览模式启动脚本

cd "$(dirname "$0")"

echo "Starting Baby Sleep Supervisor with preview..."
/usr/bin/python3 main.py
