import cv2
import numpy as np
import os

# 创建一个简单的测试图片
test_image = np.zeros((480, 640, 3), dtype=np.uint8)
test_image[:, :] = [255, 255, 255]  # 白色背景

# 添加一些颜色块
cv2.rectangle(test_image, (100, 100), (200, 200), (0, 0, 255), -1)  # 红色方块
cv2.rectangle(test_image, (300, 200), (400, 300), (0, 255, 0), -1)  # 绿色方块
cv2.rectangle(test_image, (200, 300), (300, 400), (255, 0, 0), -1)  # 蓝色方块

# 保存到当前目录
test_path = "test_image.jpg"

cv2.imwrite(test_path, test_image)
print(f"测试图片已保存到: {test_path}")
print(f"文件是否存在: {os.path.exists(test_path)}")