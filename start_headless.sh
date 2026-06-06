#!/bin/bash
# 无头后台模式启动脚本（和kid_supervisor完全一致）

cd "$(dirname "$0")"

echo "Starting Baby Sleep Supervisor in headless mode..."
/usr/bin/python3 main.py --no-preview
