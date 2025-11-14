import cv2
import numpy as np
import time
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import threading
from datetime import datetime
import argparse
from pathlib import Path

class YOLODetector:
    def __init__(self, model_path=None, config_path=None, classes_path=None, confidence_threshold=0.5, nms_threshold=0.4):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型权重文件路径
            config_path: YOLO配置文件路径
            classes_path: 类别文件路径
            confidence_threshold: 置信度阈值
            nms_threshold: 非极大值抑制阈值
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # 设置默认模型路径
        if model_path is None:
            model_path = "yolov4.weights"
        if config_path is None:
            config_path = "yolov4.cfg"
        if classes_path is None:
            classes_path = "coco.names"
        
        self.model_path = model_path
        self.config_path = config_path
        self.classes_path = classes_path
        
        # 中间层数据
        self.intermediate_outputs = []
        self.layer_names = []
        
        # 加载类别名称
        self.classes = self._load_classes(classes_path)
        
        # 加载模型
        self.net = self._load_model()
        
        # 为不同类别设置颜色
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def _load_classes(self, classes_path):
        """加载类别名称"""
        try:
            if os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
            else:
                # 默认COCO类别
                classes = [
                    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                    "toothbrush"
                ]
                print(f"警告: 未找到类别文件 {classes_path}，使用默认COCO类别")
        except Exception as e:
            print(f"加载类别文件失败: {e}")
            classes = [
                "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush"
            ]
        
        return classes
    
    def _load_model(self):
        """加载YOLO模型，增加错误处理"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                print(f"错误: 模型文件不存在: {self.model_path}")
                return None
            
            if not os.path.exists(self.config_path):
                print(f"错误: 配置文件不存在: {self.config_path}")
                return None
            
            print(f"正在加载模型... (模型: {self.model_path}, 配置: {self.config_path})")
            
            # 尝试加载模型
            net = cv2.dnn.readNetFromDarknet(self.config_path, self.model_path)
            
            # 检查模型是否成功加载
            if net is None or net.empty():
                print("错误: 模型加载失败")
                return None
            
            print("模型加载成功!")
            return net
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("请确保已下载YOLO模型文件")
            print("模型文件可以从以下链接获取:")
            print("YOLOv4: https://github.com/AlexeyAB/darknet")
            print("或者使用预训练模型:")
            print("wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights")
            print("wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg")
            return None
    
    def detect(self, image, get_intermediates=False):
        """
        对单张图片进行目标检测
        
        Args:
            image: 输入图像
            get_intermediates: 是否获取中间层数据
            
        Returns:
            检测结果列表
        """
        print(f"DEBUG: 开始检测，self.net={self.net}")
        if self.net is None:
            print("错误: 模型未加载")
            return []
        
        if image is None:
            print("错误: 输入图像为空")
            return []
        
        try:
            height, width = image.shape[:2]
            
            # 创建blob对象
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            
            # 设置输入
            self.net.setInput(blob)
            
            # 获取输出层名称
            try:
                layer_names = self.net.getLayerNames()
                # 修复OpenCV版本兼容性问题
                unconnected_layers = self.net.getUnconnectedOutLayers()
                if len(unconnected_layers.shape) == 2:
                    # 对于OpenCV 4.x版本
                    output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
                else:
                    # 对于OpenCV 3.x版本
                    output_layers = [layer_names[i - 1] for i in unconnected_layers]
                self.layer_names = layer_names  # 保存层名称
            except Exception as e:
                # 如果获取层名失败，使用默认输出层
                print(f"获取输出层失败，使用默认配置: {e}")
                output_layers = ['yolo_82', 'yolo_94', 'yolo_106']  # YOLOv4的典型输出层
            
            # 获取输出
            outputs = self.net.forward(output_layers)
            
            # 保存中间层数据（如果需要）
            if get_intermediates:
                self.intermediate_outputs = outputs
            
            # 解析检测结果
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    # 确保detection是有效的数组
                    if len(detection) < 6:
                        continue
                    
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > self.confidence_threshold:
                        # 转换坐标
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # 转换为左上角坐标
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # 非极大值抑制
            if len(boxes) > 0:
                try:
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
                    
                    results = []
                    if len(indices) > 0:
                        for i in indices.flatten():
                            if i < len(boxes):  # 确保索引有效
                                x, y, w, h = boxes[i]
                                confidence = confidences[i]
                                class_id = class_ids[i]
                                
                                results.append({
                                    'bbox': [x, y, w, h],
                                    'confidence': confidence,
                                    'class_id': class_id,
                                    'class_name': self.classes[class_id] if class_id < len(self.classes) else f"Class {class_id}"
                                })
                    return results
                    
                except Exception as e:
                    print(f"NMS处理失败: {e}")
                    # 如果NMS失败，返回所有检测结果
                    results = []
                    for i in range(len(boxes)):
                        x, y, w, h = boxes[i]
                        confidence = confidences[i]
                        class_id = class_ids[i]
                        
                        results.append({
                            'bbox': [x, y, w, h],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.classes[class_id] if class_id < len(self.classes) else f"Class {class_id}"
                        })
                    return results
            else:
                return []
                
        except Exception as e:
            print(f"检测过程中出错: {e}")
            return []
    
    def get_intermediate_data(self):
        """
        获取中间层数据
        
        Returns:
            中间层数据列表
        """
        return self.intermediate_outputs
    
    def visualize_intermediate_layers(self, image=None, save_path=None):
        """
        可视化中间层特征图
        
        Args:
            image: 输入图像（如果为None，则使用当前图像）
            save_path: 保存路径（可选）
            
        Returns:
            可视化图像
        """
        try:
            # 如果没有提供图像，则尝试获取中间层数据
            if image is not None:
                # 通过检测获取中间层数据
                self.detect(image, get_intermediates=True)
            
            # 检查是否有中间层数据
            if not hasattr(self, 'intermediate_outputs') or self.intermediate_outputs is None:
                print("警告: 没有可用的中间层数据")
                return None
            
            outputs = self.intermediate_outputs
            
            # 创建一个大的画布来显示所有特征图
            canvas_height = 0
            canvas_width = 0
            layer_images = []
            
            # 处理每个输出层
            for i, output in enumerate(outputs):
                if len(output.shape) < 3:
                    print(f"跳过第{i}层，形状无效: {output.shape}")
                    continue
                
                # 获取特征图数量
                if len(output.shape) == 4:
                    channels = output.shape[1]  # NCHW格式
                    feature_maps = output[0]  # 取第一个样本
                else:
                    channels = output.shape[2]  # NHWC格式
                    feature_maps = output[0]  # 取第一个样本
                
                # 选择前12个通道进行可视化（3x4网格）
                num_channels = min(channels, 12)
                
                # 创建网格布局
                grid_rows = 3
                grid_cols = 4
                cell_size = 64  # 每个特征图的大小
                
                # 创建画布
                grid_image = np.zeros((grid_rows * cell_size, grid_cols * cell_size, 3), dtype=np.uint8)
                
                # 填充特征图
                for j in range(num_channels):
                    if len(feature_maps.shape) == 3:
                        if feature_maps.shape[0] == channels:  # NCHW
                            feature_map = feature_maps[j]
                        else:  # NHWC
                            feature_map = feature_maps[:, :, j]
                    else:
                        feature_map = feature_maps
                    
                    # 归一化特征图
                    if np.max(feature_map) > 0:
                        feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
                        feature_map = (feature_map * 255).astype(np.uint8)
                    
                    # 调整大小
                    feature_map_resized = cv2.resize(feature_map, (cell_size, cell_size))
                    
                    # 转换为彩色图像
                    if len(feature_map_resized.shape) == 2:
                        feature_map_colored = cv2.cvtColor(feature_map_resized, cv2.COLOR_GRAY2BGR)
                    else:
                        feature_map_colored = feature_map_resized
                    
                    # 计算位置
                    row = j // grid_cols
                    col = j % grid_cols
                    
                    # 放置到网格中
                    y1 = row * cell_size
                    y2 = y1 + cell_size
                    x1 = col * cell_size
                    x2 = x1 + cell_size
                    
                    grid_image[y1:y2, x1:x2] = feature_map_colored
                
                layer_images.append(grid_image)
                
                # 更新画布大小
                canvas_height = max(canvas_height, grid_image.shape[0])
                canvas_width += grid_image.shape[1]
            
            if not layer_images:
                print("警告: 没有有效的特征图可以显示")
                return None
            
            # 创建最终的画布
            final_canvas = np.zeros((canvas_height + 50, canvas_width, 3), dtype=np.uint8)  # 额外空间用于标题
            
            # 将所有层的特征图放在画布上
            x_offset = 0
            for i, layer_img in enumerate(layer_images):
                h, w = layer_img.shape[:2]
                final_canvas[50:50+h, x_offset:x_offset+w] = layer_img
                x_offset += w
            
            # 添加标题
            cv2.putText(final_canvas, f"Intermediate Layers Visualization ({len(layer_images)} layers)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 保存图像（如果指定了路径）
            if save_path:
                cv2.imwrite(save_path, final_canvas)
                print(f"中间层可视化图像已保存到: {save_path}")
            
            return final_canvas
            
        except Exception as e:
            print(f"可视化中间层时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_image(self, image_path, save_path=None):
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            save_path: 保存路径
            
        Returns:
            处理后的图像和检测结果
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            print(f"正在处理图片: {image_path}")
            print(f"图片尺寸: {image.shape}")
            
            # 检测
            results = self.detect(image)
            
            print(f"检测到 {len(results)} 个目标")
            
            # 可视化
            output_image = self.draw_detections(image, results)
            
            # 保存结果
            if save_path:
                cv2.imwrite(save_path, output_image)
                print(f"结果已保存到: {save_path}")
            
            return output_image, results
            
        except Exception as e:
            print(f"处理图片时出错: {e}")
            return None, []
    
    def process_video(self, video_path, output_path=None, show_progress=True):
        """
        处理视频文件
        
        Args:
            video_path: 视频路径
            output_path: 输出视频路径
            show_progress: 是否显示进度
            
        Returns:
            性能报告
        """
        if self.net is None:
            print("错误: 模型未加载，无法处理视频")
            return None
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"视频信息: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
            
            # 初始化输出视频
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 性能统计
            frame_count = 0
            total_time = 0
            detection_results = defaultdict(list)
            
            print("开始处理视频...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 记录开始时间
                start_time = time.time()
                
                # 检测
                results = self.detect(frame)
                
                # 记录结束时间
                end_time = time.time()
                processing_time = end_time - start_time
                total_time += processing_time
                
                # 统计检测结果
                for result in results:
                    detection_results[result['class_name']].append(1)
                
                # 可视化
                output_frame = self.draw_detections(frame, results)
                
                # 显示进度
                if show_progress and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    avg_fps = frame_count / total_time if total_time > 0 else 0
                    print(f"进度: {progress:.1f}% | 平均FPS: {avg_fps:.2f} | 检测到 {len(results)} 个目标")
                
                # 写入输出视频
                if output_path:
                    out.write(output_frame)
                
                # 显示结果
                cv2.imshow('YOLO Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 释放资源
            cap.release()
            if output_path:
                out.release()
            cv2.destroyAllWindows()
            
            # 生成性能报告
            avg_processing_time = total_time / frame_count if frame_count > 0 else 0
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            report = {
                'video_info': {
                    'path': video_path,
                    'total_frames': total_frames,
                    'fps': fps,
                    'resolution': f"{width}x{height}"
                },
                'performance': {
                    'total_processing_time': total_time,
                    'average_processing_time': avg_processing_time,
                    'average_fps': avg_fps,
                    'frame_count': frame_count
                },
                'detection_results': dict(detection_results),
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            print(f"处理视频时出错: {e}")
            return None
    
    def process_camera(self, camera_id=0, save_path=None):
        """
        处理摄像头输入
        
        Args:
            camera_id: 摄像头ID
            save_path: 保存路径
        """
        if self.net is None:
            print("错误: 模型未加载，无法处理摄像头")
            return None
        
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                raise ValueError(f"无法打开摄像头: {camera_id}")
            
            # 初始化输出视频
            if save_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
            frame_count = 0
            total_time = 0
            
            print("摄像头检测启动，按 'q' 键退出")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 记录开始时间
                start_time = time.time()
                
                # 检测
                results = self.detect(frame)
                
                # 记录结束时间
                end_time = time.time()
                processing_time = end_time - start_time
                total_time += processing_time
                
                # 可视化
                output_frame = self.draw_detections(frame, results)
                
                # 写入输出视频
                if save_path:
                    out.write(output_frame)
                
                # 显示结果
                cv2.imshow('Camera YOLO Detection', output_frame)
                
                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 释放资源
            cap.release()
            if save_path:
                out.release()
            cv2.destroyAllWindows()
            
            # 生成性能报告
            avg_processing_time = total_time / frame_count if frame_count > 0 else 0
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            report = {
                'camera_info': {
                    'id': camera_id,
                    'frame_count': frame_count
                },
                'performance': {
                    'total_processing_time': total_time,
                    'average_processing_time': avg_processing_time,
                    'average_fps': avg_fps
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            print(f"处理摄像头时出错: {e}")
            return None
    
    def draw_detections(self, image, detections):
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果列表
            
        Returns:
            绘制后的图像
        """
        if image is None:
            return None
            
        output_image = image.copy()
        
        for detection in detections:
            try:
                x, y, w, h = detection['bbox']
                confidence = detection['confidence']
                class_id = detection['class_id']
                class_name = detection['class_name']
                
                # 绘制边界框
                color = self.colors[class_id % len(self.colors)]  # 确保颜色索引有效
                cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
                
                # 绘制标签
                label = f"{class_name}: {confidence:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output_image, (x, y - label_height - baseline - 10), 
                             (x + label_width, y), color, -1)
                cv2.putText(output_image, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception as e:
                print(f"绘制检测结果时出错: {e}")
                continue
        
        return output_image
    
    def generate_report(self, report_data, save_path=None):
        """
        生成性能报告
        
        Args:
            report_data: 报告数据
            save_path: 保存路径
        """
        try:
            # 创建报告内容
            report_content = f"""
YOLO目标检测性能报告
====================

生成时间: {report_data['timestamp']}

"""
            
            if 'video_info' in report_data:
                video_info = report_data['video_info']
                report_content += f"视频信息:\n"
                report_content += f"  路径: {video_info['path']}\n"
                report_content += f"  总帧数: {video_info['total_frames']}\n"
                report_content += f"  FPS: {video_info['fps']}\n"
                report_content += f"  分辨率: {video_info['resolution']}\n\n"
            
            if 'camera_info' in report_data:
                camera_info = report_data['camera_info']
                report_content += f"摄像头信息:\n"
                report_content += f"  ID: {camera_info['id']}\n"
                report_content += f"  处理帧数: {camera_info['frame_count']}\n\n"
            
            performance = report_data['performance']
            report_content += f"性能统计:\n"
            report_content += f"  总处理时间: {performance['total_processing_time']:.2f} 秒\n"
            report_content += f"  平均处理时间: {performance['average_processing_time']:.4f} 秒\n"
            report_content += f"  平均FPS: {performance['average_fps']:.2f}\n"
            report_content += f"  处理帧数: {performance['frame_count']}\n\n"
            
            if 'detection_results' in report_data:
                report_content += f"检测结果统计:\n"
                for class_name, count in report_data['detection_results'].items():
                    report_content += f"  {class_name}: {len(count)} 次\n"
            
            # 保存报告
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
            
            print(report_content)
            return report_content
            
        except Exception as e:
            print(f"生成报告时出错: {e}")
            return ""

class YOLOApplication:
    def __init__(self, model_path=None, config_path=None, classes_path=None):
        """
        初始化YOLO应用
        
        Args:
            model_path: 模型权重文件路径
            config_path: 模型配置文件路径
            classes_path: 类别文件路径
        """
        self.detector = YOLODetector(model_path, config_path, classes_path)
    
    def run_image_detection(self, image_path, output_path=None):
        """运行图片检测"""
        print(f"处理图片: {image_path}")
        try:
            if self.detector.net is None:
                print("错误: 模型未加载，请检查模型文件路径")
                return []
            
            output_image, results = self.detector.process_image(image_path, output_path)
            if output_image is not None:
                print(f"检测到 {len(results)} 个目标")
                
                # 显示结果
                if len(results) > 0:
                    cv2.imshow('Detection Result', output_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                return results
            else:
                print("处理失败")
                return []
        except Exception as e:
            print(f"处理图片时出错: {e}")
            return []
    
    def run_video_detection(self, video_path, output_path=None):
        """运行视频检测"""
        print(f"处理视频: {video_path}")
        try:
            if self.detector.net is None:
                print("错误: 模型未加载，请检查模型文件路径")
                return None
            
            report = self.detector.process_video(video_path, output_path)
            if report:
                report_path = f"{Path(video_path).stem}_report.txt"
                self.detector.generate_report(report, report_path)
                print(f"性能报告已保存到: {report_path}")
            return report
        except Exception as e:
            print(f"处理视频时出错: {e}")
            return None
    
    def run_camera_detection(self, camera_id=0, output_path=None):
        """运行摄像头检测"""
        print(f"启动摄像头检测 (ID: {camera_id})")
        try:
            if self.detector.net is None:
                print("错误: 模型未加载，请检查模型文件路径")
                return None
            
            report = self.detector.process_camera(camera_id, output_path)
            if report:
                report_path = f"camera_report_{camera_id}.txt"
                self.detector.generate_report(report, report_path)
                print(f"性能报告已保存到: {report_path}")
            return report
        except Exception as e:
            print(f"处理摄像头时出错: {e}")
            return None

def create_sample_image():
    """创建示例图片用于测试"""
    try:
        # 创建一个简单的测试图片
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 添加一些几何形状作为测试对象
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)  # 绿色矩形
        cv2.circle(img, (300, 150), 50, (0, 0, 255), -1)  # 红色圆形
        cv2.rectangle(img, (400, 200), (500, 300), (255, 0, 0), -1)  # 蓝色矩形
        
        cv2.imwrite('sample.jpg', img)
        print("示例图片已创建: sample.jpg")
        return 'sample.jpg'
    except Exception as e:
        print(f"创建示例图片失败: {e}")
        return None

def create_sample_video():
    """创建示例视频用于测试"""
    try:
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
        print("示例视频已创建: sample.mp4")
        return 'sample.mp4'
    except Exception as e:
        print(f"创建示例视频失败: {e}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO目标检测应用')
    parser.add_argument('--type', choices=['image', 'video', 'camera'], 
                       required=True, help='处理类型')
    parser.add_argument('--input', required=True, help='输入文件路径或摄像头ID')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--model', help='模型权重文件路径')
    parser.add_argument('--config', help='模型配置文件路径')
    parser.add_argument('--classes', help='类别文件路径')
    
    args = parser.parse_args()
    
    # 创建应用实例
    app = YOLOApplication(args.model, args.config, args.classes)
    
    if args.type == 'image':
        app.run_image_detection(args.input, args.output)
    elif args.type == 'video':
        app.run_video_detection(args.input, args.output)
    elif args.type == 'camera':
        try:
            camera_id = int(args.input)
            app.run_camera_detection(camera_id, args.output)
        except ValueError:
            print("摄像头ID必须是数字")

# 检查和下载模型的辅助函数
def check_model_files():
    """检查模型文件是否存在"""
    model_files = ["yolov4.weights", "yolov4.cfg", "coco.names"]
    missing_files = [f for f in model_files if not os.path.exists(f)]
    
    if missing_files:
        print("警告: 以下模型文件缺失:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n请下载相应的YOLO模型文件:")
        print("1. YOLOv4权重文件: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights")
        print("2. YOLOv4配置文件: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg")
        print("3. COCO类别文件: https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names")
        return False
    else:
        print("所有模型文件都已就绪")
        return True

def demo_usage():
    """演示使用方法"""
    print("=== YOLO目标检测应用演示 ===")
    print("使用前请确保:")
    print("1. 已下载YOLO模型文件")
    print("2. 安装了所有依赖包")
    print("\n基本使用方法:")
    print("python yolo_app.py --type image --input sample.jpg --output result.jpg")
    print("python yolo_app.py --type video --input sample.mp4 --output result.mp4")
    print("python yolo_app.py --type camera --input 0 --output camera_result.mp4")
    
    # 创建示例文件
    create_sample_image()
    create_sample_video()

if __name__ == "__main__":
    print("YOLO目标检测应用启动...")
    
    # 检查模型文件
    if not check_model_files():
        print("\n请下载模型文件后重新运行")
        demo_usage()
    else:
        print("模型文件检查通过，可以开始使用")
        # 运行主程序
        main()
