#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

# 测试中文路径处理
chinese_path = r"C:\Users\Peach\Pictures\P40备份241125\52489454930343084801711974734470.jpg"

print(f"原始路径: {chinese_path}")
print(f"文件是否存在: {os.path.exists(chinese_path)}")

# 方法1: 使用numpy读取文件，然后用cv2.imdecode解码
print("\n方法1: 使用numpy读取文件，然后用cv2.imdecode解码")
try:
    # 以二进制模式读取文件
    with open(chinese_path, 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    
    # 使用imdecode解码图像
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is not None:
        print("使用imdecode读取成功")
        print(f"图像尺寸: {image.shape}")
    else:
        print("使用imdecode读取失败")
except Exception as e:
    print(f"使用imdecode读取异常: {e}")

print("\n测试完成")