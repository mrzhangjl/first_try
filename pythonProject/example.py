import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 保留用于数据可视化
import threading
import time
import os
from datetime import datetime
import sys
import gc
from PIL import Image, ImageTk  # 添加PIL导入
# 尝试导入python-vlc库用于更好的音频播放
try:
    import vlc
    vlc_available = True
except ImportError:
    print("未安装python-vlc库，将尝试使用ffmpeg或pygame播放音频")
    vlc_available = False
# 添加pygame库用于音频播放
try:
    import pygame
    pygame.init()
    pygame.mixer.init()
    pygame_available = True
except ImportError:
    print("未安装pygame库，无法播放音频")
    pygame_available = False

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

class YOLO5DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO5 物体识别可视化工具")
        self.root.geometry("1400x850")
        self.root.minsize(1200, 700)
        
        # 设置主题样式
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0", font=("SimHei", 10))
        self.style.configure("TButton", font=("SimHei", 10))
        self.style.configure("TScale", background="#f0f0f0")
        self.style.configure("TNotebook", background="#f0f0f0")
        
        # 初始化变量
        self.model = None
        self.cap = None
        self.running = False
        self.detection_thread = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_det = 1000
        self.image_size = 640
        self.classes_to_detect = None  # 全部类别
        self.results = None
        self.detection_history = []
        self.frame_count = 0
        self.detection_count = 0  # 实际进行检测的帧数
        self.start_time = 0
        self.fps = 0
        self.detected_objects = {}
        # 性能优化参数
        self.frame_skip = 1  # 每frame_skip帧进行一次检测
        self.ui_update_interval = 0.1  # UI更新间隔(秒)
        self.last_ui_update_time = 0
        self.max_fps = 30  # 默认最大FPS限制
        self.colors = None
        self.class_names = []
        # 音频相关变量
        self.audio_playing = False
        self.audio_thread = None
        self.play_audio = pygame_available  # 是否播放音频
        self.audio_file = None
        self.frame_interval = 1.0 / self.max_fps  # 帧间隔时间
        
        # 创建界面
        self.create_widgets()
        
        # 加载YOLO5模型
        self.load_model_default()
    
    def create_widgets(self):
        """创建GUI界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建标签页控件
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建各个标签页
        self.create_detection_tab()
        self.create_parameter_tab()
        self.create_visualization_tab()
        self.create_optimization_tab()
        self.create_help_tab()
    
    def create_detection_tab(self):
        """创建检测标签页"""
        detection_tab = ttk.Frame(self.notebook)
        self.notebook.add(detection_tab, text="物体检测")
        
        # 左侧控制面板
        control_frame = ttk.Frame(detection_tab, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 模型加载部分
        ttk.Label(control_frame, text="模型加载", font=("SimHei", 12, "bold")).pack(pady=10)
        
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.model_path_var = tk.StringVar(value="yolov5s.pt")
        ttk.Entry(model_frame, textvariable=self.model_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(model_frame, text="浏览", command=self.browse_model).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(control_frame, text="加载模型", command=self.load_model).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(control_frame, text="使用默认模型", command=self.load_model_default).pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Separator(control_frame, orient="horizontal").pack(fill=tk.X, pady=10)
        
        # 输入源选择
        ttk.Label(control_frame, text="输入源", font=("SimHei", 12, "bold")).pack(pady=10)
        
        source_frame = ttk.LabelFrame(control_frame, text="选择输入源")
        source_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.source_var = tk.StringVar(value="image")
        ttk.Radiobutton(source_frame, text="图片", variable=self.source_var, value="image").pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(source_frame, text="视频", variable=self.source_var, value="video").pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(source_frame, text="摄像头", variable=self.source_var, value="camera").pack(anchor=tk.W, padx=10, pady=5)
        
        # 文件选择按钮
        self.file_path_var = tk.StringVar(value="")
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Entry(file_frame, textvariable=self.file_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="选择文件", command=self.browse_file).pack(side=tk.RIGHT, padx=5)
        
        # 摄像头选择
        cam_frame = ttk.Frame(control_frame)
        cam_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(cam_frame, text="摄像头ID:").pack(side=tk.LEFT, padx=5)
        self.camera_id_var = tk.IntVar(value=0)
        ttk.Combobox(cam_frame, textvariable=self.camera_id_var, values=[0, 1, 2, 3]).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Separator(control_frame, orient="horizontal").pack(fill=tk.X, pady=10)
        
        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="开始检测", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="停止检测", command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="保存结果", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Separator(control_frame, orient="horizontal").pack(fill=tk.X, pady=10)
        
        # 状态信息
        ttk.Label(control_frame, text="状态信息", font=("SimHei", 12, "bold")).pack(pady=10)
        
        status_frame = ttk.LabelFrame(control_frame, text="当前状态")
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W, padx=10, pady=5)
        
        self.fps_var = tk.StringVar(value="FPS: 0")
        ttk.Label(status_frame, textvariable=self.fps_var).pack(anchor=tk.W, padx=10, pady=5)
        
        self.objects_var = tk.StringVar(value="检测物体: 0")
        ttk.Label(status_frame, textvariable=self.objects_var).pack(anchor=tk.W, padx=10, pady=5)
        
        # 右侧显示区域
        display_frame = ttk.Frame(detection_tab)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建图形显示 - 使用Tkinter Canvas代替Matplotlib以提高性能
        self.canvas = tk.Canvas(display_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 存储图像引用以防止被垃圾回收
        self.tk_image = None
        
        # 初始显示提示信息 - 使用像素坐标而非相对坐标
        self.canvas.update_idletasks()  # 确保获取正确的画布尺寸
        canvas_width = max(800, self.canvas.winfo_width())  # 默认宽度
        canvas_height = max(600, self.canvas.winfo_height())  # 默认高度
        self.canvas.create_text(canvas_width // 2, canvas_height // 2, 
                              text="请加载模型并选择输入源开始检测", 
                              font=('SimHei', 12), fill='white', anchor='center')
    
    def create_parameter_tab(self):
        """创建参数调节标签页"""
        param_tab = ttk.Frame(self.notebook)
        self.notebook.add(param_tab, text="参数调节")
        
        # 参数框架
        param_frame = ttk.Frame(param_tab)
        param_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 检测参数
        detection_param_frame = ttk.LabelFrame(param_frame, text="检测参数", padding=10)
        detection_param_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 置信度阈值
        conf_frame = ttk.Frame(detection_param_frame)
        conf_frame.pack(fill=tk.X, pady=5)
        ttk.Label(conf_frame, text="置信度阈值:").pack(side=tk.LEFT, padx=10)
        self.conf_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                                  command=self.update_conf_threshold)
        self.conf_scale.set(0.25)
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.conf_value_var = tk.StringVar(value="0.25")
        ttk.Label(conf_frame, textvariable=self.conf_value_var, width=5).pack(side=tk.LEFT, padx=10)
        
        # IOU阈值
        iou_frame = ttk.Frame(detection_param_frame)
        iou_frame.pack(fill=tk.X, pady=5)
        ttk.Label(iou_frame, text="IOU阈值:").pack(side=tk.LEFT, padx=10)
        self.iou_scale = ttk.Scale(iou_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, 
                                 command=self.update_iou_threshold)
        self.iou_scale.set(0.45)
        self.iou_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.iou_value_var = tk.StringVar(value="0.45")
        ttk.Label(iou_frame, textvariable=self.iou_value_var, width=5).pack(side=tk.LEFT, padx=10)
        
        # 最大检测数量
        max_det_frame = ttk.Frame(detection_param_frame)
        max_det_frame.pack(fill=tk.X, pady=5)
        ttk.Label(max_det_frame, text="最大检测数量:").pack(side=tk.LEFT, padx=10)
        self.max_det_var = tk.IntVar(value=1000)
        ttk.Entry(max_det_frame, textvariable=self.max_det_var, width=10).pack(side=tk.LEFT, padx=10)
        ttk.Button(max_det_frame, text="应用", command=self.update_max_det).pack(side=tk.LEFT, padx=10)
        
        # 图像大小
        img_size_frame = ttk.Frame(detection_param_frame)
        img_size_frame.pack(fill=tk.X, pady=5)
        ttk.Label(img_size_frame, text="图像大小:").pack(side=tk.LEFT, padx=10)
        self.img_size_var = tk.IntVar(value=640)
        ttk.Combobox(img_size_frame, textvariable=self.img_size_var, 
                    values=[320, 416, 512, 640, 736, 832], width=8).pack(side=tk.LEFT, padx=10)
        ttk.Button(img_size_frame, text="应用", command=self.update_img_size).pack(side=tk.LEFT, padx=10)
        
        # 性能优化参数
        perf_param_frame = ttk.LabelFrame(param_frame, text="性能优化参数", padding=10)
        perf_param_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 帧跳过设置
        skip_frame_frame = ttk.Frame(perf_param_frame)
        skip_frame_frame.pack(fill=tk.X, pady=5)
        ttk.Label(skip_frame_frame, text="检测间隔帧数:").pack(side=tk.LEFT, padx=10)
        self.frame_skip_var = tk.IntVar(value=1)
        ttk.Combobox(skip_frame_frame, textvariable=self.frame_skip_var, 
                    values=[1, 2, 3, 4, 5], width=5).pack(side=tk.LEFT, padx=10)
        ttk.Button(skip_frame_frame, text="应用", command=self.update_frame_skip).pack(side=tk.LEFT, padx=10)
        ttk.Label(skip_frame_frame, text="(1=每帧检测, 2=每隔1帧检测一次, 以此类推)", font=('SimHei', 9)).pack(side=tk.LEFT, padx=10)
        
        # FPS限制设置
        fps_limit_frame = ttk.Frame(perf_param_frame)
        fps_limit_frame.pack(fill=tk.X, pady=5)
        ttk.Label(fps_limit_frame, text="最大FPS限制:").pack(side=tk.LEFT, padx=10)
        self.max_fps_var = tk.IntVar(value=self.max_fps)
        ttk.Combobox(fps_limit_frame, textvariable=self.max_fps_var, 
                    values=list(range(1, 121)), width=5).pack(side=tk.LEFT, padx=10)
        ttk.Button(fps_limit_frame, text="应用", command=self.update_max_fps).pack(side=tk.LEFT, padx=10)
        ttk.Label(fps_limit_frame, text="限制检测的最大帧率，范围1-120", font=('SimHei', 9)).pack(side=tk.LEFT, padx=10)
        
        # 音频控制
        audio_frame = ttk.Frame(perf_param_frame)
        audio_frame.pack(fill=tk.X, pady=5)
        
        if pygame_available:
            ttk.Label(audio_frame, text="音频控制:").pack(side=tk.LEFT, padx=10)
            self.audio_check_var = tk.BooleanVar(value=True)
            audio_checkbox = ttk.Checkbutton(audio_frame, text="播放视频音频", 
                                           variable=self.audio_check_var,
                                           command=self.toggle_audio)
            audio_checkbox.pack(side=tk.LEFT, padx=10)
            ttk.Label(audio_frame, text="注意: 音频播放可能会略微增加系统资源占用", font=('SimHei', 9)).pack(side=tk.LEFT, padx=10)
        else:
            ttk.Label(audio_frame, text="未安装pygame库，无法播放音频", foreground="red").pack(side=tk.LEFT, padx=10)
            ttk.Label(audio_frame, text="请运行: pip install pygame 来启用音频功能", font=('SimHei', 9)).pack(side=tk.LEFT, padx=10)
        
        # 类别过滤
        class_filter_frame = ttk.LabelFrame(param_frame, text="类别过滤", padding=10)
        class_filter_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(class_filter_frame, text="选择要检测的类别 (留空表示全部):").pack(anchor=tk.W, padx=10, pady=5)
        
        self.class_listbox = tk.Listbox(class_filter_frame, selectmode=tk.MULTIPLE, width=30, height=15)
        self.class_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = ttk.Scrollbar(class_filter_frame, orient=tk.VERTICAL, command=self.class_listbox.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y, pady=5)
        self.class_listbox.config(yscrollcommand=scrollbar.set)
        
        button_frame = ttk.Frame(class_filter_frame)
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        
        ttk.Button(button_frame, text="全选", command=self.select_all_classes).pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="取消全选", command=self.deselect_all_classes).pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="应用", command=self.apply_class_filter).pack(fill=tk.X, pady=5)
    
    def create_visualization_tab(self):
        """创建数据可视化标签页"""
        viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(viz_tab, text="数据可视化")
        
        # 顶部控制
        control_frame = ttk.Frame(viz_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(control_frame, text="选择可视化类型:", font=("SimHei", 10)).pack(side=tk.LEFT, padx=10)
        
        self.viz_type_var = tk.StringVar(value="class_dist")
        viz_options = ["类别分布", "检测频率", "置信度分布", "检测速度"]
        viz_values = ["class_dist", "detection_freq", "conf_dist", "speed"]
        
        for i, (option, value) in enumerate(zip(viz_options, viz_values)):
            ttk.Radiobutton(control_frame, text=option, variable=self.viz_type_var, 
                          value=value, command=self.update_visualization).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(control_frame, text="刷新", command=self.update_visualization).pack(side=tk.RIGHT, padx=10)
        
        # 可视化显示区域
        viz_frame = ttk.Frame(viz_tab)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.viz_fig, self.viz_ax = plt.subplots(figsize=(10, 6), dpi=100)
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=viz_frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始显示提示信息
        self.viz_ax.text(0.5, 0.5, "开始检测后将显示可视化数据", 
                        ha="center", va="center", fontsize=12, transform=self.viz_ax.transAxes)
        self.viz_canvas.draw()
        
        # 底部信息
        info_frame = ttk.LabelFrame(viz_tab, text="统计信息", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.total_frames_var = tk.StringVar(value="总帧数: 0")
        self.total_objects_var = tk.StringVar(value="总检测物体: 0")
        self.avg_fps_var = tk.StringVar(value="平均FPS: 0.0")
        
        ttk.Label(info_frame, textvariable=self.total_frames_var).pack(side=tk.LEFT, padx=20)
        ttk.Label(info_frame, textvariable=self.total_objects_var).pack(side=tk.LEFT, padx=20)
        ttk.Label(info_frame, textvariable=self.avg_fps_var).pack(side=tk.LEFT, padx=20)
    
    def create_optimization_tab(self):
        """创建优化和学习方法标签页"""
        opt_tab = ttk.Frame(self.notebook)
        self.notebook.add(opt_tab, text="优化学习")
        
        # 创建笔记本控件来组织不同的优化方法
        opt_notebook = ttk.Notebook(opt_tab)
        opt_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 性能优化
        perf_frame = ttk.Frame(opt_notebook)
        opt_notebook.add(perf_frame, text="性能优化")
        
        perf_text = scrolledtext.ScrolledText(perf_frame, wrap=tk.WORD, font=("SimHei", 10))
        perf_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        perf_content = """YOLOv5 性能优化方法：

1. 模型选择：
   - 使用较小的模型如 yolov5s.pt 可获得更快的推理速度
   - 根据精度需求选择合适的模型大小 (s < m < l < x)

2. 输入尺寸优化：
   - 降低图像分辨率 (320x320) 可提高速度但降低精度
   - 提高图像分辨率 (832x832) 可提高精度但降低速度

3. 批量推理：
   - 使用 batch_size > 1 进行批量推理可提高吞吐量
   - 在GPU上效果更明显

4. 硬件加速：
   - 使用CUDA GPU加速推理
   - 考虑使用TensorRT进行模型转换和优化
   - 对于边缘设备，可使用ONNX Runtime或TensorFlow Lite

5. 模型剪枝和量化：
   - 模型剪枝：移除不重要的权重减少模型大小
   - 量化：将32位浮点参数转换为8位整数，可加速推理

6. 非最大抑制(NMS)参数调整：
   - 适当提高IOU阈值可减少重复检测
   - 调整置信度阈值可平衡精度和召回率

7. 前处理优化：
   - 使用更快的图像加载和预处理方法
   - 考虑使用多线程进行图像预处理
        """
        perf_text.insert(tk.END, perf_content)
        perf_text.config(state=tk.DISABLED)
        
        # 训练技巧
        train_frame = ttk.Frame(opt_notebook)
        opt_notebook.add(train_frame, text="训练技巧")
        
        train_text = scrolledtext.ScrolledText(train_frame, wrap=tk.WORD, font=("SimHei", 10))
        train_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        train_content = """YOLOv5 训练优化技巧：

1. 数据准备：
   - 确保数据集标注准确且一致
   - 数据集应包含各种场景、角度和光照条件
   - 建议使用至少100张图像进行训练，每类至少20张

2. 数据增强：
   - 旋转、缩放、翻转、裁剪等几何变换
   - 亮度、对比度、饱和度调整
   - 马赛克增强(Mosaic)：YOLOv5默认使用
   - 混合增强(MixUp)：提高模型鲁棒性

3. 超参数调整：
   - 学习率：建议使用余弦退火学习率调度器
   - 权重衰减：防止过拟合，默认为0.0005
   - 批次大小：根据GPU内存调整，通常32-64

4. 迁移学习：
   - 使用预训练权重加速收敛
   - 冻结早期层进行微调
   - 对于小数据集特别有效

5. 损失函数优化：
   - 调整分类、定位和置信度损失的权重
   - 考虑使用Focal Loss处理类别不平衡

6. 模型评估：
   - 使用mAP(平均精度)评估模型性能
   - 在验证集上监控训练进度
   - 使用混淆矩阵分析错误类型

7. 避免过拟合：
   - 增加数据增强强度
   - 使用早停(Early Stopping)
   - 应用Dropout或正则化技术
        """
        train_text.insert(tk.END, train_content)
        train_text.config(state=tk.DISABLED)
        
        # 部署指南
        deploy_frame = ttk.Frame(opt_notebook)
        opt_notebook.add(deploy_frame, text="部署指南")
        
        deploy_text = scrolledtext.ScrolledText(deploy_frame, wrap=tk.WORD, font=("SimHei", 10))
        deploy_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        deploy_content = """YOLOv5 模型部署指南：

1. 模型导出：
   - 导出为ONNX格式：python export.py --weights yolov5s.pt --include onnx
   - 导出为TensorRT：需要额外安装TensorRT
   - 导出为TorchScript：python export.py --weights yolov5s.pt --include torchscript

2. 不同平台部署：
   - 服务器部署：使用Flask/FastAPI创建REST API
   - 移动设备：使用TFLite或CoreML
   - 嵌入式设备：使用ONNX Runtime或TensorRT

3. 实时推理优化：
   - 使用多线程处理输入/输出
   - 实现预处理和推理的并行化
   - 考虑使用异步推理模式

4. 批量处理：
   - 实现批处理队列提高吞吐量
   - 动态批量大小调整

5. 内存管理：
   - 及时释放不必要的张量
   - 使用内存池减少内存分配开销
   - 对于长时间运行的应用，定期进行垃圾回收

6. 监控和维护：
   - 实现性能监控（FPS、延迟等）
   - 设置日志系统记录错误和警告
   - 提供模型更新机制
        """
        deploy_text.insert(tk.END, deploy_content)
        deploy_text.config(state=tk.DISABLED)
    
    def create_help_tab(self):
        """创建帮助标签页"""
        help_tab = ttk.Frame(self.notebook)
        self.notebook.add(help_tab, text="使用帮助")
        
        help_text = scrolledtext.ScrolledText(help_tab, wrap=tk.WORD, font=("SimHei", 10))
        help_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        help_content = """YOLO5 物体识别可视化工具 - 使用指南

1. 基本操作流程：
   - 加载YOLO模型（可使用默认模型或自定义模型）
   - 选择输入源（图片、视频或摄像头）
   - 调整检测参数（可选）
   - 点击"开始检测"按钮

2. 输入源说明：
   - 图片：选择本地图片文件进行单张检测
   - 视频：选择本地视频文件进行连续检测
   - 摄像头：使用电脑摄像头进行实时检测

3. 参数调节说明：
   - 置信度阈值：控制检测结果的可信度，值越高结果越可信但可能漏检
   - IOU阈值：控制非最大抑制的严格程度，影响重叠物体的检测
   - 最大检测数量：限制每张图片/帧的最大检测物体数
   - 图像大小：调整输入到模型的图像分辨率，影响检测精度和速度
   - 类别过滤：可选择只检测特定类别的物体

4. 数据可视化：
   - 类别分布：显示各类型物体的检测数量统计
   - 检测频率：显示随时间变化的检测数量趋势
   - 置信度分布：显示检测结果的置信度分布情况
   - 检测速度：显示FPS随时间的变化

5. 保存结果：
   - 检测过程中可随时点击"保存结果"按钮
   - 结果将保存为图片文件，包含检测框和标签

6. 注意事项：
   - 首次加载模型可能需要一些时间，请耐心等待
   - 视频和摄像头检测时，可点击"停止检测"随时中断
   - 建议在GPU环境下运行以获得更好的性能
   - 对于大型视频文件，处理可能需要较长时间

7. 系统要求：
   - Python 3.8+
   - PyTorch 1.7+
   - OpenCV
   - Matplotlib
   - 推荐使用CUDA兼容的GPU以加速推理
        """
        help_text.insert(tk.END, help_content)
        help_text.config(state=tk.DISABLED)
    
    def load_model_default(self):
        """加载默认的YOLOv5模型"""
        try:
            self.status_var.set("正在加载默认模型...")
            self.root.update_idletasks()
            
            # 尝试加载本地模型，如果不存在则下载
            model_path = "yolov5s.pt"
            if not os.path.exists(model_path):
                self.status_var.set("下载默认模型...")
                self.root.update_idletasks()
                # 这里可以添加下载逻辑
                messagebox.showwarning("警告", "未找到默认模型文件，请先下载yolov5s.pt或使用浏览按钮选择模型文件")
                self.status_var.set("就绪")
                return
            
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            self.model.max_det = self.max_det
            
            # 设置为GPU模式（如果可用）
            if torch.cuda.is_available():
                self.model.cuda()
                self.status_var.set("默认模型加载成功 (GPU)")
            else:
                self.model.cpu()
                self.status_var.set("默认模型加载成功 (CPU)")
            
            # 获取类别名称和设置颜色
            self.class_names = self.model.names
            self.update_class_listbox()
            self.generate_colors()
            
            messagebox.showinfo("成功", "默认模型加载成功")
            
        except Exception as e:
            self.status_var.set(f"模型加载失败: {str(e)}")
            messagebox.showerror("错误", f"加载模型时出错: {str(e)}")
    
    def load_model(self):
        """加载指定的YOLOv5模型"""
        try:
            model_path = self.model_path_var.get()
            if not model_path or not os.path.exists(model_path):
                messagebox.showerror("错误", "模型文件不存在，请选择有效的模型文件")
                return
            
            self.status_var.set("正在加载模型...")
            self.root.update_idletasks()
            
            # 卸载之前的模型以释放内存
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            self.model.conf = self.conf_threshold
            self.model.iou = self.iou_threshold
            self.model.max_det = self.max_det
            
            # 设置为GPU模式（如果可用）
            if torch.cuda.is_available():
                self.model.cuda()
                self.status_var.set("模型加载成功 (GPU)")
            else:
                self.model.cpu()
                self.status_var.set("模型加载成功 (CPU)")
            
            # 获取类别名称和设置颜色
            self.class_names = self.model.names
            self.update_class_listbox()
            self.generate_colors()
            
            messagebox.showinfo("成功", "模型加载成功")
            
        except Exception as e:
            self.status_var.set(f"模型加载失败: {str(e)}")
            messagebox.showerror("错误", f"加载模型时出错: {str(e)}")
    
    def browse_model(self):
        """浏览并选择模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择YOLO模型文件",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
    
    def browse_file(self):
        """浏览并选择图片或视频文件"""
        source_type = self.source_var.get()
        if source_type == "image":
            file_types = [("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif"), ("所有文件", "*.*")]
            title = "选择图片文件"
        else:  # video
            file_types = [("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
            title = "选择视频文件"
        
        file_path = filedialog.askopenfilename(title=title, filetypes=file_types)
        if file_path:
            self.file_path_var.set(file_path)
    
    def update_conf_threshold(self, value):
        """更新置信度阈值"""
        self.conf_threshold = float(value)
        self.conf_value_var.set(f"{self.conf_threshold:.2f}")
        if self.model is not None:
            self.model.conf = self.conf_threshold
    
    def update_iou_threshold(self, value):
        """更新IOU阈值"""
        self.iou_threshold = float(value)
        self.iou_value_var.set(f"{self.iou_threshold:.2f}")
        if self.model is not None:
            self.model.iou = self.iou_threshold
    
    def update_max_det(self):
        """更新最大检测数量"""
        try:
            self.max_det = self.max_det_var.get()
            if self.model is not None:
                self.model.max_det = self.max_det
            messagebox.showinfo("成功", f"最大检测数量已更新为 {self.max_det}")
        except Exception as e:
            messagebox.showerror("错误", f"更新最大检测数量时出错: {str(e)}")
    
    def update_img_size(self):
        """更新图像大小"""
        try:
            self.image_size = self.img_size_var.get()
            messagebox.showinfo("成功", f"图像大小已更新为 {self.image_size}")
        except Exception as e:
            messagebox.showerror("错误", f"更新图像大小时出错: {str(e)}")
    
    def update_frame_skip(self):
        """更新帧跳过参数"""
        try:
            self.frame_skip = self.frame_skip_var.get()
            messagebox.showinfo("成功", f"检测间隔帧数已更新为 {self.frame_skip + 1}")
        except Exception as e:
            messagebox.showerror("错误", f"更新帧跳过参数时出错: {str(e)}")
    
    def update_max_fps(self):
        """更新最大FPS限制"""
        try:
            self.max_fps = self.max_fps_var.get()
            # 更新帧间隔时间
            self.frame_interval = 1.0 / self.max_fps
            messagebox.showinfo("成功", f"最大FPS限制已更新为 {self.max_fps}")
        except Exception as e:
            messagebox.showerror("错误", f"更新FPS限制时出错: {str(e)}")
    
    def update_class_listbox(self):
        """更新类别列表框"""
        self.class_listbox.delete(0, tk.END)
        for i, name in self.class_names.items():
            self.class_listbox.insert(tk.END, f"{i}: {name}")
    
    def select_all_classes(self):
        """全选类别"""
        for i in range(self.class_listbox.size()):
            self.class_listbox.selection_set(i)
    
    def deselect_all_classes(self):
        """取消全选类别"""
        self.class_listbox.selection_clear(0, tk.END)
    
    def apply_class_filter(self):
        """应用类别过滤"""
        selected_indices = self.class_listbox.curselection()
        if not selected_indices:
            self.classes_to_detect = None  # 检测所有类别
            messagebox.showinfo("成功", "已设置为检测所有类别")
        else:
            # 获取选中的类别ID
            self.classes_to_detect = [int(self.class_listbox.get(i).split(':')[0]) for i in selected_indices]
            messagebox.showinfo("成功", f"已设置为检测 {len(self.classes_to_detect)} 个类别")
    
    def generate_colors(self):
        """为每个类别生成颜色"""
        import random
        random.seed(42)  # 设置随机种子以确保颜色一致
        self.colors = {}
        for i in range(len(self.class_names)):
            self.colors[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def start_detection(self):
        """开始检测"""
        if self.model is None:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        if self.running:
            messagebox.showinfo("提示", "检测已在运行中")
            return
        
        try:
            # 更新UI状态
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED)
            self.running = True
            self.frame_count = 0
            self.detection_count = 0
            self.start_time = time.time()
            self.last_ui_update_time = self.start_time
            self.frame_start_time = self.start_time  # 用于计算实际FPS
            self.detection_history = []
            self.detected_objects = {}
            
            # 根据输入源开始检测
            source_type = self.source_var.get()
            
            if source_type == "image":
                file_path = self.file_path_var.get()
                if not file_path or not os.path.exists(file_path):
                    messagebox.showerror("错误", "请选择有效的图片文件")
                    self.reset_ui_state()
                    return
                self.detect_image(file_path)
            else:
                # 视频或摄像头使用线程处理
                self.detection_thread = threading.Thread(target=self.detect_stream)
                self.detection_thread.daemon = True
                self.detection_thread.start()
                
        except Exception as e:
            self.status_var.set(f"检测启动失败: {str(e)}")
            messagebox.showerror("错误", f"启动检测时出错: {str(e)}")
            self.reset_ui_state()
    
    def stop_detection(self):
        """停止检测"""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # 停止音频播放
        self.stop_audio_playback()
        
        self.reset_ui_state()
        self.status_var.set("检测已停止")
    
    def toggle_audio(self):
        """切换音频播放状态"""
        self.play_audio = self.audio_check_var.get()
        if self.running and self.source_var.get() == "video":
            if self.play_audio and not self.audio_playing:
                self.start_audio_playback()
            elif not self.play_audio and self.audio_playing:
                self.stop_audio_playback()
    
    def start_audio_playback(self):
        """开始播放视频音频 - 优化版"""
        if not self.audio_file or self.audio_playing:
            return
        
        def audio_playback_thread():
            self.audio_playing = True
            player = None
            process = None
            
            try:
                print(f"尝试播放音频: {self.audio_file}")
                
                # 直接使用ffmpeg作为首选方案，这是最可靠的跨平台视频音频播放方式
                try:
                    import subprocess
                    print("使用ffmpeg播放音频")
                    # 使用更简单的ffplay命令，确保音频设备正确初始化
                    cmd = [
                        'ffplay', '-i', self.audio_file, '-nodisp', '-autoexit',
                        '-loglevel', 'quiet', '-af', 'volume=1.0'
                    ]
                    # 使用shell=True以确保在Windows系统上能正确找到ffplay
                    process = subprocess.Popen(cmd, shell=True)
                    
                    # 等待进程结束或用户停止
                    while self.running and self.play_audio:
                        if process.poll() is not None:
                            break
                        time.sleep(0.1)
                    
                except Exception as ffmpeg_error:
                    print(f"ffmpeg播放失败: {str(ffmpeg_error)}")
                    # 如果ffmpeg不可用，尝试其他方案
                    
                    # 使用python-vlc
                    if vlc_available:
                        print("使用python-vlc播放音频")
                        try:
                            # 创建VLC实例，不使用--no-video参数，让VLC自动处理
                            instance = vlc.Instance()
                            player = instance.media_player_new()
                            media = instance.media_new(self.audio_file)
                            player.set_media(media)
                            
                            # 显式设置音频输出
                            player.audio_set_volume(100)
                            
                            # 开始播放
                            player.play()
                            
                            # 检查播放状态
                            start_time = time.time()
                            while self.running and self.play_audio:
                                state = player.get_state()
                                if state == vlc.State.Ended or state == vlc.State.Error:
                                    break
                                # 播放5秒后检查是否有音频输出
                                if time.time() - start_time > 5 and player.audio_get_volume() == 0:
                                    print("尝试调整VLC音频设置")
                                    player.audio_set_volume(100)
                                time.sleep(0.1)
                        except Exception as vlc_error:
                            print(f"VLC播放失败: {str(vlc_error)}")
                    
                    # 最后尝试pygame（但它不能直接播放视频文件）
                    elif pygame_available:
                        print("pygame无法直接播放视频文件中的音频")
                        # 尝试提取音频临时文件（需要ffmpeg）
                        try:
                            temp_audio = "temp_audio.wav"
                            extract_cmd = [
                                'ffmpeg', '-i', self.audio_file, '-y', '-vn', '-acodec', 'pcm_s16le',
                                '-ar', '44100', '-ac', '2', temp_audio
                            ]
                            print(f"尝试提取音频到临时文件: {temp_audio}")
                            extract_process = subprocess.Popen(extract_cmd, shell=True)
                            extract_process.wait(timeout=5)
                            
                            # 如果成功提取，尝试播放
                            if os.path.exists(temp_audio) and os.path.getsize(temp_audio) > 0:
                                pygame.mixer.init(frequency=44100, size=-16, channels=2)
                                pygame.mixer.music.load(temp_audio)
                                pygame.mixer.music.set_volume(1.0)
                                pygame.mixer.music.play()
                                
                                while self.running and self.play_audio and pygame.mixer.music.get_busy():
                                    time.sleep(0.1)
                        except Exception as pygame_error:
                            print(f"音频提取/播放失败: {str(pygame_error)}")
                        finally:
                            # 清理临时文件
                            if os.path.exists(temp_audio):
                                try:
                                    os.remove(temp_audio)
                                except:
                                    pass
                    
                    else:
                        print("无法播放音频：请安装ffmpeg")
                        # 在Windows上，尝试使用内置的媒体播放器
                        try:
                            print("尝试使用Windows媒体播放器")
                            os.startfile(self.audio_file)
                            time.sleep(2)  # 给播放器一些启动时间
                        except:
                            print("Windows媒体播放器启动失败")
                    
            except Exception as e:
                print(f"音频播放错误: {str(e)}")
            finally:
                self.audio_playing = False
                
                # 确保停止播放
                if process and process.poll() is None:
                    try:
                        process.terminate()
                        process.wait(timeout=0.5)
                    except:
                        pass
                
                if player and vlc_available:
                    try:
                        player.stop()
                    except:
                        pass
                
                if pygame_available:
                    try:
                        pygame.mixer.music.stop()
                        pygame.mixer.quit()
                    except:
                        pass
        
        # 创建并启动音频线程
        self.audio_thread = threading.Thread(target=audio_playback_thread, daemon=True)
        self.audio_thread.start()
    
    def stop_audio_playback(self):
        """停止音频播放"""
        self.play_audio = False
        self.audio_playing = False
        
        # 停止pygame音频
        if pygame_available:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            except:
                pass
        
        # 等待音频线程结束
        if hasattr(self, 'audio_thread') and self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=0.5)
    
    def can_use_ffmpeg(self):
        """检查系统是否安装了ffmpeg"""
        try:
            import subprocess
            subprocess.run(['ffplay', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def reset_ui_state(self):
        """重置UI状态"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL if self.results is not None else tk.DISABLED)
        self.running = False
    
    def detect_image(self, image_path):
        """检测单张图片"""
        try:
            self.status_var.set("正在处理图片...")
            self.root.update_idletasks()
            
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("无法读取图片文件")
            
            # 执行检测
            results = self.model(img, size=self.image_size)
            self.results = results
            
            # 处理检测结果
            self.process_detection_results(results, img)
            
            # 更新状态
            self.status_var.set("图片检测完成")
            self.save_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.status_var.set(f"图片检测失败: {str(e)}")
            messagebox.showerror("错误", f"处理图片时出错: {str(e)}")
            self.reset_ui_state()
    
    def detect_stream(self):
        """检测视频流或摄像头"""
        try:
            source_type = self.source_var.get()
            
            # 打开视频源
            if source_type == "video":
                file_path = self.file_path_var.get()
                if not file_path or not os.path.exists(file_path):
                    raise Exception("请选择有效的视频文件")
                self.cap = cv2.VideoCapture(file_path)
                # 设置视频读取的缓冲区大小
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # 尝试播放视频音频
                if pygame_available and self.play_audio:
                    self.audio_file = file_path
                    self.start_audio_playback()
            else:  # camera
                camera_id = self.camera_id_var.get()
                self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                raise Exception("无法打开视频源")
            
            self.status_var.set(f"正在进行{source_type}检测...")
            
            # 检测循环
            prev_detection_results = None  # 保存上一次检测结果用于帧跳过
            detection_start_time = time.time()
            display_frame_count = 0  # 显示帧数计数
            display_start_time = time.time()  # 显示计时开始
            
            # 优化视频读取，对于视频文件设置合适的帧率
            if source_type == "video":
                # 获取视频原始帧率
                original_fps = self.cap.get(cv2.CAP_PROP_FPS)
                # 计算适当的帧率，避免过高或过低
                target_fps = min(self.max_fps, original_fps) if original_fps > 0 else self.max_fps
                # 设置合适的间隔时间，避免过频繁地读取视频帧
                self.frame_interval = 1.0 / target_fps
            
            while self.running:
                start_time = time.time()
                
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    if source_type == "video":
                        # 视频播放完毕
                        break
                    else:
                        # 摄像头读取失败，继续尝试
                        continue
                
                self.frame_count += 1
                current_time = time.time()
                
                # 根据帧跳过参数决定是否进行检测
                perform_detection = (self.frame_count % (self.frame_skip + 1) == 0)
                
                if perform_detection:
                    # 执行检测
                    results = self.model(frame, size=self.image_size)
                    self.results = results
                    prev_detection_results = results
                    self.detection_count += 1
                    
                    # 计算检测FPS
                    detection_elapsed_time = time.time() - detection_start_time
                    self.fps = self.detection_count / detection_elapsed_time if detection_elapsed_time > 0 else 0
                    detection_start_time = time.time()
                    
                    # 处理检测结果并更新UI (如果需要)
                    if current_time - self.last_ui_update_time >= self.ui_update_interval:
                        # 传递一个较小的图像副本给UI更新，减少内存占用
                        h, w = frame.shape[:2]
                        if h > 720 or w > 1280:  # 如果图像太大，先缩小再传递
                            scale = min(720/h, 1280/w)
                            small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                            self.process_detection_results(results, small_frame)
                        else:
                            self.process_detection_results(results, frame.copy())
                        self.last_ui_update_time = current_time
                else:
                    # 使用上一次的检测结果，但更新帧
                    if prev_detection_results is not None:
                        # 只在需要时更新UI
                        if current_time - self.last_ui_update_time >= self.ui_update_interval:
                            # 传递一个较小的图像副本给UI更新，减少内存占用
                            h, w = frame.shape[:2]
                            if h > 720 or w > 1280:  # 如果图像太大，先缩小再传递
                                scale = min(720/h, 1280/w)
                                small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                                self.process_detection_results(prev_detection_results, small_frame)
                            else:
                                self.process_detection_results(prev_detection_results, frame.copy())
                            self.last_ui_update_time = current_time
                
                display_frame_count += 1
                display_elapsed_time = time.time() - display_start_time
                
                # 计算实际显示的FPS
                if display_elapsed_time > 0:
                    actual_fps = display_frame_count / display_elapsed_time
                else:
                    actual_fps = 0
                
                # 优化的帧率控制逻辑，减少卡顿
                frame_processing_time = time.time() - start_time
                
                # 计算应该等待的时间
                target_sleep_time = max(0, self.frame_interval - frame_processing_time)
                
                # 使用更精确的时间控制，避免过度睡眠
                if target_sleep_time > 0:
                    # 对于较大的睡眠需求，使用分段睡眠
                    if target_sleep_time > 0.01:  # 大于10ms时
                        # 先睡眠大部分时间
                        time.sleep(target_sleep_time * 0.85)
                        # 精确等待剩余时间
                        precise_start = time.time()
                        while time.time() - precise_start < target_sleep_time * 0.15:
                            pass
                
                # 周期性更新FPS显示，显示实际FPS
                if self.frame_count % 5 == 0:
                    self.root.after(0, lambda: self.fps_var.set(f"FPS: {actual_fps:.1f}"))
                
                # 每30秒重置一次计数，避免数值溢出
                if display_elapsed_time > 30:
                    display_frame_count = 0
                    display_start_time = time.time()
                
        except Exception as e:
            error_msg = f"流检测失败: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("错误", f"处理视频流时出错: {str(e)}"))
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.root.after(0, self.reset_ui_state)
    
    def process_detection_results(self, results, frame):
        """处理检测结果并显示"""
        # 转换结果
        df = results.pandas().xyxy[0]  # 获取检测结果DataFrame
        
        # 应用类别过滤
        if self.classes_to_detect is not None:
            df = df[df['class'].isin(self.classes_to_detect)]
        
        # 复制帧用于绘制
        img_with_boxes = frame.copy()
        
        # 绘制检测框
        for _, row in df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            cls_id = int(row['class'])
            cls_name = row['name']
            
            # 绘制边界框
            color = self.colors.get(cls_id, (255, 0, 0))
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img_with_boxes, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 更新检测统计 (仅在实际检测帧更新)
            if self.frame_count % (self.frame_skip + 1) == 0:
                if cls_name not in self.detected_objects:
                    self.detected_objects[cls_name] = 0
                self.detected_objects[cls_name] += 1
        
        # 更新检测历史 (仅在实际检测帧更新)
        if self.frame_count % (self.frame_skip + 1) == 0:
            current_time = time.time() - self.start_time
            self.detection_history.append({
                'time': current_time,
                'objects': len(df),
                'fps': self.fps
            })
        
        # 更新UI显示
        self.root.after(0, lambda: self.update_display(img_with_boxes, len(df)))
        
        # 更新统计信息 (降低更新频率)
        if self.frame_count % 10 == 0:
            self.root.after(0, self.update_statistics)
    
    def update_display(self, frame, num_objects):
        """更新显示界面 - 使用PIL和Tkinter Canvas代替Matplotlib以提高性能"""
        try:
            # 获取画布尺寸
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # 如果画布尺寸有效，进行显示
            if canvas_width > 1 and canvas_height > 1:
                # 转换BGR为RGB (一次性操作)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 计算图像调整比例，保持纵横比
                img_height, img_width = rgb_frame.shape[:2]
                ratio = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                # 调整图像大小 - 进一步优化
                rgb_frame = cv2.resize(rgb_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)  # 使用INTER_AREA更适合缩小
                
                # 转换为PIL图像并转换为Tkinter可用的图像
                pil_image = Image.fromarray(rgb_frame)
                self.tk_image = ImageTk.PhotoImage(image=pil_image)
                
                # 清除画布并显示图像（居中显示）
                self.canvas.delete("all")
                x_pos = (canvas_width - new_width) // 2
                y_pos = (canvas_height - new_height) // 2
                self.canvas.create_image(x_pos, y_pos, anchor='nw', image=self.tk_image)
            
            # 更新状态信息 - 减少字符串格式化频率
            if num_objects != getattr(self, '_last_num_objects', -1):
                self.objects_var.set(f"检测物体: {num_objects}")
                self._last_num_objects = num_objects
        except Exception as e:
            # 异常处理，避免UI崩溃
            if getattr(self, '_last_error', '') != str(e):  # 避免重复打印相同错误
                print(f"更新显示时出错: {str(e)}")
                self._last_error = str(e)
    
    def update_statistics(self):
        """更新统计信息"""
        total_time = time.time() - self.start_time
        # 计算实际检测FPS而非总帧数FPS
        avg_detection_fps = self.detection_count / total_time if total_time > 0 else 0
        
        self.total_frames_var.set(f"总帧数: {self.frame_count}")
        total_objects = sum(self.detected_objects.values())
        self.total_objects_var.set(f"总检测物体: {total_objects}")
        self.avg_fps_var.set(f"平均FPS: {avg_detection_fps:.1f}")
    
    def update_visualization(self):
        """更新可视化图表"""
        viz_type = self.viz_type_var.get()
        
        self.viz_ax.clear()
        
        if not self.detection_history:
            self.viz_ax.text(0.5, 0.5, "暂无检测数据", 
                           ha="center", va="center", fontsize=12, transform=self.viz_ax.transAxes)
            self.viz_canvas.draw()
            return
        
        try:
            if viz_type == "class_dist":
                # 类别分布饼图
                if self.detected_objects:
                    labels = list(self.detected_objects.keys())
                    sizes = list(self.detected_objects.values())
                    
                    # 只显示前10个类别以避免图表过于拥挤
                    if len(labels) > 10:
                        top_labels = labels[:9]
                        top_sizes = sizes[:9]
                        other_size = sum(sizes[9:])
                        top_labels.append("其他")
                        top_sizes.append(other_size)
                        labels, sizes = top_labels, top_sizes
                    
                    self.viz_ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    self.viz_ax.axis('equal')  # 确保饼图是圆的
                    self.viz_ax.set_title('检测物体类别分布')
            
            elif viz_type == "detection_freq":
                # 检测频率折线图 - 优化数据点数量
                times = [h['time'] for h in self.detection_history]
                objects = [h['objects'] for h in self.detection_history]
                
                # 如果数据点太多，进行降采样显示
                if len(times) > 100:
                    # 保留最近100个数据点
                    times = times[-100:]
                    objects = objects[-100:]
                
                self.viz_ax.plot(times, objects, 'b-', linewidth=2)
                self.viz_ax.set_xlabel('时间 (秒)')
                self.viz_ax.set_ylabel('检测物体数量')
                self.viz_ax.set_title('检测频率随时间变化')
                self.viz_ax.grid(True, alpha=0.3)
            
            elif viz_type == "conf_dist":
                # 置信度分布图
                if self.results:
                    df = self.results.pandas().xyxy[0]
                    if not df.empty:
                        confidences = df['confidence'].values
                        self.viz_ax.hist(confidences, bins=20, alpha=0.7, color='green')
                        self.viz_ax.set_xlabel('置信度')
                        self.viz_ax.set_ylabel('频率')
                        self.viz_ax.set_title('检测结果置信度分布')
                        self.viz_ax.grid(True, alpha=0.3)
                    else:
                        self.viz_ax.text(0.5, 0.5, "当前帧无检测结果", 
                                       ha="center", va="center", fontsize=12, transform=self.viz_ax.transAxes)
                else:
                    self.viz_ax.text(0.5, 0.5, "暂无检测结果", 
                                   ha="center", va="center", fontsize=12, transform=self.viz_ax.transAxes)
            
            elif viz_type == "speed":
                # 检测速度折线图 - 优化数据点数量
                times = [h['time'] for h in self.detection_history]
                fps = [h['fps'] for h in self.detection_history]
                
                # 如果数据点太多，进行降采样显示
                if len(times) > 100:
                    # 保留最近100个数据点
                    times = times[-100:]
                    fps = fps[-100:]
                
                self.viz_ax.plot(times, fps, 'r-', linewidth=2)
                self.viz_ax.set_xlabel('时间 (秒)')
                self.viz_ax.set_ylabel('FPS')
                self.viz_ax.set_title('检测速度随时间变化')
                self.viz_ax.grid(True, alpha=0.3)
            
            self.viz_fig.tight_layout()
            self.viz_canvas.draw()
            
        except Exception as e:
            self.viz_ax.text(0.5, 0.5, f"可视化更新失败: {str(e)}", 
                           ha="center", va="center", fontsize=10, transform=self.viz_ax.transAxes)
            self.viz_canvas.draw()
    
    def save_results(self):
        """保存检测结果"""
        if not self.results:
            messagebox.showinfo("提示", "暂无检测结果可保存")
            return
        
        try:
            # 创建保存目录
            save_dir = "detection_results"
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"detection_{timestamp}.jpg")
            
            # 保存图像
            img = self.results.render()[0]  # 获取渲染后的图像
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # 保存检测数据到CSV
            csv_filename = os.path.join(save_dir, f"detection_data_{timestamp}.csv")
            df = self.results.pandas().xyxy[0]
            df.to_csv(csv_filename, index=False)
            
            messagebox.showinfo("成功", f"检测结果已保存到:\n{filename}\n{csv_filename}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存结果时出错: {str(e)}")

    def on_closing(self):
        """窗口关闭时的处理"""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=0.5)
        if self.cap:
            self.cap.release()
        self.root.destroy()

def main():
    """主函数"""
    root = tk.Tk()
    app = YOLO5DetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()