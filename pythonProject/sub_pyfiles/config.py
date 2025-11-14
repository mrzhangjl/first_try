# config.py - 配置文件
import os


class Config:
    # 模型配置
    MODEL_PATH = "yolov4.weights"
    CONFIG_PATH = "yolov4.cfg"
    CLASSES_PATH = "coco.names"

    # 检测参数
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4

    # 输出配置
    OUTPUT_DIR = "output"
    REPORT_DIR = "reports"

    # 性能优化配置
    CUDA_ENABLE = True
    THREAD_COUNT = 4

    # 默认输入输出路径
    DEFAULT_IMAGE = "sample.jpg"
    DEFAULT_VIDEO = "sample.mp4"

    @classmethod
    def ensure_directories(cls):
        """确保必要的目录存在"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.REPORT_DIR, exist_ok=True)


# utils.py - 工具函数
import cv2
import numpy as np
from pathlib import Path


def create_sample_image():
    """创建示例图片用于测试"""
    # 创建一个简单的测试图片
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # 添加一些几何形状作为测试对象
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)  # 绿色矩形
    cv2.circle(img, (300, 150), 50, (0, 0, 255), -1)  # 红色圆形
    cv2.rectangle(img, (400, 200), (500, 300), (255, 0, 0), -1)  # 蓝色矩形

    cv2.imwrite('sample.jpg', img)
    return 'sample.jpg'


def create_sample_video():
    """创建示例视频用于测试"""
    # 创建一个简单的测试视频
    width, height = 640, 480
    fps = 30
    duration = 5  # 5秒

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('sample.mp4', fourcc, fps, (width, height))

    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 移动的矩形
        x = int((i * 5) % (width - 100))
        y = int((i * 3) % (height - 100))

        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)
        cv2.rectangle(frame, (x + 20, y + 20), (x + 80, y + 80), (255, 0, 0), -1)

        out.write(frame)

    out.release()
    return 'sample.mp4'


# performance_monitor.py - 性能监控器
import time
import psutil
import threading
from collections import deque


class PerformanceMonitor:
    def __init__(self):
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.start_time = time.time()

    def update_metrics(self):
        """更新性能指标"""
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent

        self.cpu_usage.append(cpu)
        self.memory_usage.append(memory)

    def get_stats(self):
        """获取统计信息"""
        if len(self.cpu_usage) == 0:
            return None

        return {
            'cpu_avg': sum(self.cpu_usage) / len(self.cpu_usage),
            'cpu_max': max(self.cpu_usage),
            'memory_avg': sum(self.memory_usage) / len(self.memory_usage),
            'memory_max': max(self.memory_usage),
            'uptime': time.time() - self.start_time
        }


# requirements.txt - 依赖包
"""
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
psutil>=5.8.0
"""

# README.md - 使用说明
README_CONTENT = """
# YOLO目标检测应用

一个功能完整的YOLO目标检测应用，支持图片、视频和摄像头输入。

## 功能特性

- ✅ 支持图片、视频、摄像头输入
- ✅ 实时可视化检测结果
- ✅ 详细的性能报告生成
- ✅ 视频性能优化
- ✅ 多类别目标检测
- ✅ 可配置的检

## 安装依赖

```bash
pip install opencv-python numpy matplotlib psutil
"""