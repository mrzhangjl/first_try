#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os
import sys

# 测试中文路径处理
chinese_path = r"C:\Users\Peach\Pictures\P40备份241125\52489454930343084801711974734470.jpg"

print(f"原始路径: {chinese_path}")
print(f"文件是否存在: {os.path.exists(chinese_path)}")

# 方法1: 直接使用OpenCV读取
print("\n方法1: 直接使用OpenCV读取")
image1 = cv2.imread(chinese_path)
if image1 is not None:
    print("直接读取成功")
else:
    print("直接读取失败")

# 方法2: 使用UTF-8编码处理路径
print("\n方法2: 使用UTF-8编码处理路径")
try:
    utf8_path = chinese_path.encode('utf-8').decode('utf-8')
    print(f"UTF-8处理后路径: {utf8_path}")
    image2 = cv2.imread(utf8_path)
    if image2 is not None:
        print("UTF-8编码路径读取成功")
    else:
        print("UTF-8编码路径读取失败")
except Exception as e:
    print(f"UTF-8编码处理异常: {e}")

# 方法3: 使用系统默认编码
print("\n方法3: 使用系统默认编码")
try:
    sys_path = chinese_path.encode(sys.getfilesystemencoding()).decode(sys.getfilesystemencoding())
    print(f"系统编码处理后路径: {sys_path}")
    image3 = cv2.imread(sys_path)
    if image3 is not None:
        print("系统编码路径读取成功")
    else:
        print("系统编码路径读取失败")
except Exception as e:
    print(f"系统编码处理异常: {e}")

# 方法4: 检查OpenCV版本
print(f"\nOpenCV版本: {cv2.__version__}")

print("\n测试完成")