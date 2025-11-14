import cv2
import os

# 测试文件路径
test_path = r"C:/Users/Peach/Pictures/P40备份241125/52489454930343084801711974734470.jpg"
print(f"原始路径: {test_path}")

# 规范化路径
normalized_path = os.path.normpath(test_path)
print(f"规范化路径: {normalized_path}")

# 检查文件是否存在
print(f"文件是否存在: {os.path.exists(normalized_path)}")

# 尝试用OpenCV读取
print("尝试用OpenCV读取文件...")
image = cv2.imread(normalized_path)
if image is not None:
    print(f"成功读取图片，尺寸: {image.shape}")
else:
    print("无法读取图片")

# 尝试用不同的编码方式
try:
    # 使用UTF-8编码
    utf8_path = normalized_path.encode('utf-8').decode('utf-8')
    print(f"UTF-8编码路径: {utf8_path}")
    image = cv2.imread(utf8_path)
    if image is not None:
        print(f"UTF-8编码路径读取成功，尺寸: {image.shape}")
    else:
        print("UTF-8编码路径读取失败")
except Exception as e:
    print(f"UTF-8编码处理异常: {e}")