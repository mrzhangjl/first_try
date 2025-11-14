import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_app import YOLODetector
import inspect

# 获取YOLODetector类的所有方法
methods = [method[0] for method in inspect.getmembers(YOLODetector, predicate=inspect.isfunction)]
print("YOLODetector类的方法:")
for method in methods:
    print(f"  - {method}")

# 检查是否有draw_detections方法
has_draw_detections = 'draw_detections' in methods
print(f"\n是否有draw_detections方法: {has_draw_detections}")

# 创建实例并检查
try:
    detector = YOLODetector()
    print(f"\n实例是否有draw_detections方法: {hasattr(detector, 'draw_detections')}")
except Exception as e:
    print(f"\n创建实例时出错: {e}")