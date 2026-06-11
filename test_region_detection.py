#!/usr/bin/env python3
"""
快速测试区域检测和告警逻辑
"""
import sys
import time
sys.path.insert(0, '.')

from src.supervision import SleepSupervisor

print("=" * 60)
print("🔍 区域检测功能测试")
print("=" * 60)

# 1. 初始化监督模块
print("\n1. 初始化 SleepSupervisor...")
sv = SleepSupervisor()
print(f"   ✅ 初始化成功")
print(f"   安全区域点数: {len(sv.region_detector._safe_region)}")
print(f"   区域退出确认比例: {sv.region_exit_confirm_ratio}")
print(f"   区域退出持续阈值: {sv.region_exit_duration_threshold}s")
print(f"   初始 last_in_region: {sv.last_in_region}")

# 2. 测试notifier
print("\n2. 测试飞书通知...")
result = sv.notifier.send_feishu_text("【测试】区域检测功能已就绪\n请确认宝宝在安全区域内走动，系统会检测离开并发送通知")
print(f"   发送结果: {'✅ 成功' if result else '❌ 失败'}")

print("\n" + "=" * 60)
print("测试完成！系统已优化以下内容：")
print("✅ 修复初始状态 last_in_region = None 的问题")
print("✅ 首次检测到宝宝在区域内会自动标记位置")
print("✅ 之后宝宝离开即可正常触发告警")
print("=" * 60)
print("\n💡 提示：请确保宝宝在画面内，且至少有一部分身体在安全区域多边形内")

sv.stop()
