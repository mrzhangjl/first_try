import sys
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo_app import YOLODetector, YOLOApplication

class YOLOGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO目标检测工具")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # 设置样式
        self.setup_styles()
        
        # 初始化变量
        self.current_image = None
        self.current_video = None
        self.current_camera = None
        self.is_processing = False
        self.is_camera_active = False
        self.video_capture = None
        self.camera_thread = None
        self.processing_thread = None
        
        # 模型相关
        self.detector = None
        self.app = None
        
        # 创建界面
        self.create_widgets()
        
        # 初始化模型
        self.initialize_model()
        
        # 添加调试信息，查看文件路径的初始值
        # print(f"DEBUG: 初始化时 file_path_var 的值: '{self.file_path_var.get()}'")
        
        # 设置示例图片文件
        sample_image = "sample.jpg"
        if os.path.exists(sample_image):
            self.file_path_var.set(sample_image)
            self.input_type.set("image")
            # 自动开始检测（暂时注释掉以便手动测试）
            # self.start_detection()
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def initialize_model(self):
        """初始化模型"""
        print("DEBUG: 开始初始化模型")
        try:
            # 获取当前脚本所在目录的绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 构建模型文件的绝对路径
            model_path = os.path.join(current_dir, "yolov4.weights")
            config_path = os.path.join(current_dir, "yolov4.cfg")
            classes_path = os.path.join(current_dir, "coco.names")
            
            print(f"DEBUG: 模型文件路径: {model_path}")
            print(f"DEBUG: 配置文件路径: {config_path}")
            print(f"DEBUG: 类别文件路径: {classes_path}")
            
            # 初始化YOLO检测器
            self.detector = YOLODetector(
                model_path=model_path,
                config_path=config_path,
                classes_path=classes_path,
                confidence_threshold=self.confidence_var.get(),
                nms_threshold=self.nms_var.get()
            )
            print(f"DEBUG: YOLODetector对象创建完成, self.detector={self.detector}")
            print(f"DEBUG: self.detector.net={self.detector.net}")
            
            # 检查模型是否成功加载
            if self.detector.net is not None:
                print("DEBUG: 模型初始化完成")
                self.status_var.set("模型已加载")
                self.model_initialized = True  # 添加模型初始化成功的标志
            else:
                print("DEBUG: 模型初始化失败")
                self.status_var.set("模型加载失败")
                self.model_initialized = False  # 添加模型初始化失败的标志
        except Exception as e:
            print(f"DEBUG: 模型初始化异常: {e}")
            self.status_var.set(f"初始化错误: {str(e)}")
            self.model_initialized = False  # 添加模型初始化失败的标志
    
    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置自定义样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Primary.TButton', font=('Arial', 10, 'bold'), padding=6)
        style.configure('Secondary.TButton', font=('Arial', 9), padding=4)
        style.configure('TFrame', background='#ecf0f1')
        style.configure('Card.TFrame', background='white', relief='raised', borderwidth=1)
        
        # 配置进度条样式
        style.configure('Horizontal.TProgressbar', thickness=20)
    
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 标题
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="YOLO目标检测工具", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # 模型状态标签
        self.model_status_label = ttk.Label(title_frame, text="模型未加载", foreground='red')
        self.model_status_label.pack(side=tk.RIGHT)
        
        # 主要内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(content_frame, text="控制面板", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 输入源选择
        input_frame = ttk.LabelFrame(control_frame, text="输入源", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.input_type = tk.StringVar(value="image")
        ttk.Radiobutton(input_frame, text="图片", variable=self.input_type, value="image", 
                       command=self.on_input_type_change).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(input_frame, text="视频", variable=self.input_type, value="video", 
                       command=self.on_input_type_change).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(input_frame, text="摄像头", variable=self.input_type, value="camera", 
                       command=self.on_input_type_change).pack(anchor=tk.W, pady=2)
        
        # 文件选择区域
        self.file_frame = ttk.LabelFrame(control_frame, text="文件选择", padding=10)
        self.file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(self.file_frame, text="选择文件", command=self.select_file, 
                  style='Primary.TButton').pack(fill=tk.X, pady=(0, 5))
        
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(self.file_frame, textvariable=self.file_path_var, state='readonly')
        self.file_path_entry.pack(fill=tk.X)
        
        # 摄像头设置（默认隐藏）
        self.camera_frame = ttk.LabelFrame(control_frame, text="摄像头设置", padding=10)
        self.camera_frame.pack(fill=tk.X, pady=(0, 10))
        self.camera_frame.pack_forget()  # 默认隐藏
        
        ttk.Label(self.camera_frame, text="摄像头ID:").pack(anchor=tk.W)
        self.camera_id_var = tk.StringVar(value="0")
        camera_id_frame = ttk.Frame(self.camera_frame)
        camera_id_frame.pack(fill=tk.X, pady=(5, 10))
        ttk.Entry(camera_id_frame, textvariable=self.camera_id_var, width=10).pack(side=tk.LEFT)
        
        # 参数设置
        params_frame = ttk.LabelFrame(control_frame, text="检测参数", padding=10)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(params_frame, text="置信度阈值:").pack(anchor=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.confidence_var, 
                                    orient=tk.HORIZONTAL, command=self.on_confidence_change)
        confidence_scale.pack(fill=tk.X, pady=(5, 5))
        self.confidence_label = ttk.Label(params_frame, text="0.50")
        self.confidence_label.pack()
        
        ttk.Label(params_frame, text="NMS阈值:").pack(anchor=tk.W)
        self.nms_var = tk.DoubleVar(value=0.4)
        nms_scale = ttk.Scale(params_frame, from_=0.0, to=1.0, variable=self.nms_var, 
                             orient=tk.HORIZONTAL, command=self.on_nms_change)
        nms_scale.pack(fill=tk.X, pady=(5, 5))
        self.nms_label = ttk.Label(params_frame, text="0.40")
        self.nms_label.pack()
        
        # 操作按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(button_frame, text="开始检测", command=self.start_detection, 
                                      style='Primary.TButton')
        self.start_button.pack(fill=tk.X, pady=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="停止检测", command=self.stop_detection, 
                                     style='Secondary.TButton', state='disabled')
        self.stop_button.pack(fill=tk.X)
        
        # 进度条
        self.progress_frame = ttk.Frame(control_frame)
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_frame.pack_forget()
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, 
                                           maximum=100, style='Horizontal.TProgressbar')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(self.progress_frame, text="准备就绪")
        self.progress_label.pack()
        
        # 右侧显示区域
        display_frame = ttk.Frame(content_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 结果显示区域
        result_notebook = ttk.Notebook(display_frame)
        result_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 图像显示标签页
        self.image_frame = ttk.Frame(result_notebook)
        result_notebook.add(self.image_frame, text="图像显示")
        self.create_image_display()
        
        # 检测结果标签页
        self.results_frame = ttk.Frame(result_notebook)
        result_notebook.add(self.results_frame, text="检测结果")
        self.create_results_display()
        
        # 性能统计标签页
        self.performance_frame = ttk.Frame(result_notebook)
        result_notebook.add(self.performance_frame, text="性能统计")
        self.create_performance_display()
        
        # 中间层可视化标签页
        self.visualization_frame = ttk.Frame(result_notebook)
        result_notebook.add(self.visualization_frame, text="中间层可视化")
        self.create_visualization_display()
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_image_display(self):
        """创建图像显示区域"""
        # 创建画布和滚动条
        canvas_frame = ttk.Frame(self.image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_canvas = tk.Canvas(canvas_frame, bg='white')
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 绑定鼠标滚轮事件
        self.image_canvas.bind("<MouseWheel>", self.on_mousewheel)
    
    def create_results_display(self):
        """创建检测结果显示区域"""
        # 创建树形视图
        tree_frame = ttk.Frame(self.results_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建滚动条
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        
        # 创建树形视图
        self.results_tree = ttk.Treeview(tree_frame, columns=('Class', 'Confidence', 'BBox'), 
                                        show='headings', yscrollcommand=v_scrollbar.set, 
                                        xscrollcommand=h_scrollbar.set)
        
        # 配置列标题
        self.results_tree.heading('Class', text='类别')
        self.results_tree.heading('Confidence', text='置信度')
        self.results_tree.heading('BBox', text='边界框')
        
        self.results_tree.column('Class', width=100)
        self.results_tree.column('Confidence', width=100)
        self.results_tree.column('BBox', width=200)
        
        # 绑定滚动条
        v_scrollbar.config(command=self.results_tree.yview)
        h_scrollbar.config(command=self.results_tree.xview)
        
        # 布局
        self.results_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # 统计信息
        stats_frame = ttk.Frame(self.results_frame)
        stats_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        self.total_objects_label = ttk.Label(stats_frame, text="检测到的目标总数: 0")
        self.total_objects_label.pack(side=tk.LEFT)
    
    def create_performance_display(self):
        """创建性能统计显示区域"""
        # 创建matplotlib图表
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.tight_layout(pad=3.0)
        
        # 创建画布
        self.performance_canvas = FigureCanvasTkAgg(self.fig, self.performance_frame)
        self.performance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 初始化图表
        self.ax1.set_title("处理时间统计")
        self.ax1.set_xlabel("帧数")
        self.ax1.set_ylabel("时间(秒)")
        
        self.ax2.set_title("FPS统计")
        self.ax2.set_xlabel("帧数")
        self.ax2.set_ylabel("FPS")
        
        # 性能数据
        self.processing_times = []
        self.fps_values = []
    
    def create_visualization_display(self):
        """创建中间层可视化显示区域"""
        # 创建说明标签
        info_label = ttk.Label(self.visualization_frame, 
                              text="中间层可视化功能将在检测过程中显示模型的内部特征图",
                              wraplength=500, justify=tk.CENTER)
        info_label.pack(pady=10)
        
        # 创建可视化框架
        self.intermediate_frame = ttk.Frame(self.visualization_frame)
        self.intermediate_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加提示标签
        placeholder_label = ttk.Label(self.intermediate_frame, 
                                     text="检测完成后将显示中间层可视化结果",
                                     font=("Arial", 10), foreground="gray")
        placeholder_label.pack(expand=True)

    def display_intermediate_layers(self, image):
        """显示中间层可视化"""
        try:
            # 清除之前的可视化
            for widget in self.intermediate_frame.winfo_children():
                widget.destroy()
            
            # 获取中间层可视化图像
            vis_image = self.detector.visualize_intermediate_layers(image)
            
            if vis_image is not None:
                # 转换OpenCV图像为PIL图像
                rgb_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                # 调整图像大小以适应显示区域
                max_width = 800
                max_height = 600
                
                img_width, img_height = pil_image.size
                scale_x = max_width / img_width
                scale_y = max_height / img_height
                scale = min(scale_x, scale_y, 1.0)
                
                if scale < 1.0 or (img_width > max_width or img_height > max_height):
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                
                # 转换为Tkinter图像
                tk_image = ImageTk.PhotoImage(pil_image)
                
                # 创建标签显示图像
                label = tk.Label(self.intermediate_frame, image=tk_image)
                label.image = tk_image  # 保持引用
                label.pack(pady=10)
                
                print("中间层可视化显示成功")
            else:
                # 如果没有可视化图像，显示提示信息
                label = tk.Label(self.intermediate_frame, text="无中间层数据可显示", 
                               font=("Arial", 12), fg="gray")
                label.pack(pady=20)
                print("没有中间层数据可显示")
                
        except Exception as e:
            print(f"显示中间层可视化时出错: {e}")
            # 显示错误信息
            error_label = tk.Label(self.intermediate_frame, text=f"可视化错误: {str(e)}", 
                                 font=("Arial", 10), fg="red")
            error_label.pack(pady=10)
    
    def display_detection_results(self, results):
        """显示检测结果"""
        # 清除现有结果
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # 添加新结果
        for result in results:
            class_name = result.get('class_name', 'Unknown')
            confidence = result.get('confidence', 0)
            bbox = result.get('bbox', [0, 0, 0, 0])
            bbox_str = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
            
            self.results_tree.insert('', tk.END, values=(class_name, f"{confidence:.2f}", bbox_str))
        
        # 更新统计信息
        self.total_objects_label.config(text=f"检测到的目标总数: {len(results)}")
    
    def start_video_detection(self):
        """开始视频检测"""
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("错误", "请选择视频文件")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("错误", "文件不存在")
            return
        
        self.is_processing = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_var.set(0)
        self.progress_label.config(text="正在处理视频...")
        
        # 在新线程中执行检测
        self.processing_thread = threading.Thread(target=self.process_video, args=(file_path,))
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_image(self, file_path):
        """处理图片"""
        print(f"DEBUG: 开始处理图片，self.detector={self.detector}")
        try:
            # 更新模型参数
            if self.detector:
                self.detector.confidence_threshold = self.confidence_var.get()
                self.detector.nms_threshold = self.nms_var.get()
            
            # 使用numpy和imdecode读取图片以支持中文路径
            print(f"DEBUG: 处理前的文件路径: {file_path}")
            
            try:
                # 使用numpy读取文件字节
                with open(file_path, 'rb') as f:
                    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                
                # 使用imdecode解码图像
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is None:
                    print("DEBUG: 使用imdecode读取失败")
                    self.root.after(0, self.detection_error, f"无法读取图片: {file_path}")
                    return
                else:
                    print("DEBUG: 使用imdecode读取成功")
            except Exception as e:
                print(f"DEBUG: 读取文件时发生异常: {e}")
                self.root.after(0, self.detection_error, f"无法读取图片: {file_path}")
                return
            
            self.root.after(0, self.update_progress, 50, "正在处理图片...")
            
            # 记录开始时间
            start_time = time.time()
            
            # 检测
            results = self.detector.detect(image, get_intermediates=True)
            
            # 记录结束时间
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 可视化
            output_image = self.detector.draw_detections(image, results)
            
            # 计算FPS
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # 更新UI
            self.root.after(0, self.update_image_display, output_image, results, processing_time, fps)
            
            # 完成处理
            self.root.after(0, self.image_detection_completed, processing_time, fps)
                
        except Exception as e:
            self.root.after(0, self.detection_error, f"处理图片时出错: {str(e)}")
    
    def update_image_display(self, image, results, processing_time, fps):
        """更新图片显示"""
        try:
            # 转换OpenCV图像为PIL图像
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # 保存当前图像
            self.current_image = pil_image
            
            # 调整图像大小以适应显示区域
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # 计算缩放比例
                img_width, img_height = pil_image.size
                scale_x = canvas_width / img_width
                scale_y = canvas_height / img_height
                scale = min(scale_x, scale_y, 1.0)  # 不放大图像
                
                if scale < 1.0:
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为Tkinter图像
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # 清除画布
            self.image_canvas.delete("all")
            
            # 在画布中心显示图像
            canvas_width = self.image_canvas.winfo_width() or 800
            canvas_height = self.image_canvas.winfo_height() or 600
            
            x = max(0, (canvas_width - pil_image.width) // 2)
            y = max(0, (canvas_height - pil_image.height) // 2)
            
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=tk_image)
            self.image_canvas.image = tk_image  # 保持引用
            
            # 更新滚动区域
            self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))
            
            # 显示检测结果
            self.display_detection_results(results)
            
            # 显示中间层可视化
            self.display_intermediate_layers(image)
            
        except Exception as e:
            print(f"更新图片显示时出错: {e}")
    
    def image_detection_completed(self, processing_time, fps):
        """图片检测完成"""
        self.progress_var.set(100)
        self.progress_label.config(text=f"图片处理完成: {processing_time:.3f}秒, FPS: {fps:.2f}")
        self.status_var.set(f"图片检测完成，处理时间: {processing_time:.3f}秒, FPS: {fps:.2f}")
        self.finish_detection()
    
    def process_video(self, file_path):
        """处理视频"""
        print(f"DEBUG: 开始处理视频，self.detector={self.detector}")
        try:
            # 更新模型参数
            if self.detector:
                self.detector.confidence_threshold = self.confidence_var.get()
                self.detector.nms_threshold = self.nms_var.get()
            
            # 处理文件路径，确保中文字符正确编码
            print(f"DEBUG: 处理前的视频文件路径: {file_path}")
            
            # 使用正确的编码方式处理路径
            if isinstance(file_path, str):
                # 确保路径使用系统默认编码
                safe_path = file_path.encode('utf-8').decode('utf-8')
                # 再次规范化路径
                safe_path = os.path.normpath(safe_path)
            else:
                safe_path = file_path
            
            print(f"DEBUG: 处理后的安全路径: {safe_path}")
            
            # 打开视频文件
            cap = cv2.VideoCapture(safe_path)
            
            # 如果第一次打开失败，尝试用原始路径
            if not cap.isOpened():
                print("DEBUG: 第一次打开失败，尝试原始路径...")
                cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                self.root.after(0, self.detection_error, "无法打开视频文件")
                return
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"视频信息: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
            
            frame_count = 0
            start_time = time.time()
            
            # 性能统计
            self.processing_times = []
            self.fps_values = []
            
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 记录开始时间
                frame_start_time = time.time()
                
                # 检测（仅在第一帧获取中间层数据）
                get_intermediates = (frame_count == 1)
                results = self.detector.detect(frame, get_intermediates=get_intermediates)
                
                # 记录结束时间
                frame_end_time = time.time()
                processing_time = frame_end_time - frame_start_time
                
                # 计算FPS
                current_fps = 1.0 / processing_time if processing_time > 0 else 0
                
                # 保存性能数据
                self.processing_times.append(processing_time)
                self.fps_values.append(current_fps)
                
                # 可视化
                output_frame = self.detector.draw_detections(frame, results)
                
                # 更新UI（每30帧更新一次以提高性能）
                if frame_count % 30 == 0 or frame_count == 1:
                    self.root.after(0, self.update_video_frame, output_frame, results, frame_count, total_frames, frame)
                
                # 更新进度
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                self.root.after(0, self.update_progress, progress, f"处理帧 {frame_count}/{total_frames}")
                
                # 控制处理速度
                time.sleep(0.01)
            
            # 释放资源
            cap.release()
            
            # 完成处理
            if self.is_processing:
                total_time = time.time() - start_time
                avg_fps = frame_count / total_time if total_time > 0 else 0
                self.root.after(0, self.video_detection_completed, frame_count, total_time, avg_fps)
                
        except Exception as e:
            self.root.after(0, self.detection_error, f"处理视频时出错: {str(e)}")
    
    def update_video_frame(self, frame, results, frame_count, total_frames, original_frame):
        """更新视频帧显示"""
        if not self.is_processing:
            return
            
        try:
            # 转换OpenCV图像为PIL图像
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # 保存当前图像
            self.current_image = pil_image
            
            # 调整图像大小以适应显示区域
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # 计算缩放比例
                img_width, img_height = pil_image.size
                scale_x = canvas_width / img_width
                scale_y = canvas_height / img_height
                scale = min(scale_x, scale_y, 1.0)  # 不放大图像
                
                if scale < 1.0:
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # 转换为Tkinter图像
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # 清除画布
            self.image_canvas.delete("all")
            
            # 在画布中心显示图像
            canvas_width = self.image_canvas.winfo_width() or 800
            canvas_height = self.image_canvas.winfo_height() or 600
            
            x = max(0, (canvas_width - pil_image.width) // 2)
            y = max(0, (canvas_height - pil_image.height) // 2)
            
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=tk_image)
            self.image_canvas.image = tk_image  # 保持引用
            
            # 更新滚动区域
            self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))
            
            # 显示检测结果
            self.display_detection_results(results)
            
            # 显示中间层可视化（仅在第一帧）
            if frame_count == 1:
                self.display_intermediate_layers(original_frame)
            
        except Exception as e:
            print(f"更新视频帧时出错: {e}")
    
    def update_performance_chart(self):
        """更新性能图表"""
        try:
            if len(self.processing_times) > 1:
                # 清除之前的图表
                self.ax1.clear()
                self.ax2.clear()
                
                # 绘制处理时间
                x_data = list(range(len(self.processing_times)))
                self.ax1.plot(x_data, self.processing_times, 'b-', linewidth=1)
                self.ax1.set_title("处理时间统计")
                self.ax1.set_xlabel("帧数")
                self.ax1.set_ylabel("时间(秒)")
                self.ax1.grid(True, alpha=0.3)
                
                # 绘制FPS
                self.ax2.plot(x_data, self.fps_values, 'r-', linewidth=1)
                self.ax2.set_title("FPS统计")
                self.ax2.set_xlabel("帧数")
                self.ax2.set_ylabel("FPS")
                self.ax2.grid(True, alpha=0.3)
                
                # 刷新画布
                self.performance_canvas.draw()
                
        except Exception as e:
            print(f"更新性能图表时出错: {e}")
    
    def update_progress(self, progress, text):
        """更新进度"""
        if not self.is_processing:
            return
            
        self.progress_var.set(progress)
        self.progress_label.config(text=text)
    
    def video_detection_completed(self, frame_count, total_time, avg_fps):
        """视频检测完成"""
        self.progress_var.set(100)
        self.progress_label.config(text=f"视频处理完成: {frame_count} 帧, 平均FPS: {avg_fps:.2f}")
        self.status_var.set(f"视频检测完成，处理了 {frame_count} 帧，平均FPS: {avg_fps:.2f}")
        self.finish_detection()
    
    def start_camera_detection(self):
        """开始摄像头检测"""
        try:
            camera_id = int(self.camera_id_var.get())
        except ValueError:
            messagebox.showerror("错误", "摄像头ID必须是数字")
            return
        
        self.is_processing = True
        self.is_camera_active = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_var.set(0)
        self.progress_label.config(text="正在启动摄像头...")
        
        # 在新线程中执行检测
        self.camera_thread = threading.Thread(target=self.process_camera, args=(camera_id,))
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def process_camera(self, camera_id):
        """处理摄像头"""
        try:
            # 更新模型参数
            if self.detector:
                self.detector.confidence_threshold = self.confidence_var.get()
                self.detector.nms_threshold = self.nms_var.get()
            
            # 打开摄像头
            self.video_capture = cv2.VideoCapture(camera_id)
            if not self.video_capture.isOpened():
                self.root.after(0, self.detection_error, f"无法打开摄像头 {camera_id}")
                return
            
            self.root.after(0, self.update_progress, 0, "摄像头已启动")
            
            frame_count = 0
            start_time = time.time()
            
            # 性能统计
            self.processing_times = []
            self.fps_values = []
            
            while self.is_processing and self.is_camera_active:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 记录开始时间
                frame_start_time = time.time()
                
                # 检测（仅在第一帧获取中间层数据）
                get_intermediates = (frame_count == 1)
                results = self.detector.detect(frame, get_intermediates=get_intermediates)
                
                # 记录结束时间
                frame_end_time = time.time()
                processing_time = frame_end_time - frame_start_time
                
                # 计算FPS
                current_fps = 1.0 / processing_time if processing_time > 0 else 0
                
                # 保存性能数据
                self.processing_times.append(processing_time)
                self.fps_values.append(current_fps)
                
                # 可视化
                output_frame = self.detector.draw_detections(frame, results)
                
                # 更新UI（每5帧更新一次以提高性能）
                if frame_count % 5 == 0:
                    self.root.after(0, self.update_video_frame, output_frame, results, frame_count, 0, frame)
                
                # 更新进度
                elapsed_time = time.time() - start_time
                self.root.after(0, self.update_progress, 50, f"摄像头运行中... ({elapsed_time:.1f}s)")
                
                # 控制处理速度
                time.sleep(0.01)
            
            # 释放资源
            if self.video_capture:
                self.video_capture.release()
            
            # 完成处理
            if self.is_processing:
                self.root.after(0, self.camera_detection_completed)
                
        except Exception as e:
            self.root.after(0, self.detection_error, f"处理摄像头时出错: {str(e)}")
    
    def camera_detection_completed(self):
        """摄像头检测完成"""
        self.progress_label.config(text="摄像头已停止")
        self.status_var.set("摄像头检测已停止")
        self.finish_detection()
    
    def select_file(self):
        """选择文件"""
        input_type = self.input_type.get()
        if input_type == "image":
            file_types = [("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff")]
            title = "选择图片文件"
        elif input_type == "video":
            file_types = [("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv")]
            title = "选择视频文件"
        else:
            return
        
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=file_types
        )
        
        if file_path:
            print(f"DEBUG: 通过文件选择对话框设置文件路径: {file_path}")
            # 规范化文件路径
            normalized_path = os.path.normpath(file_path)
            print(f"DEBUG: 规范化后的文件路径: {normalized_path}")
            
            # 确保路径编码正确（特别是包含中文的路径）
            try:
                # 使用UTF-8编码处理路径
                if isinstance(normalized_path, str):
                    # 确保路径字符串使用正确的编码
                    normalized_path = normalized_path.encode('utf-8').decode('utf-8')
            except Exception as e:
                print(f"DEBUG: 路径编码处理异常: {e}")
            
            self.file_path_var.set(normalized_path)

    def start_detection(self):
        """开始检测"""
        print(f"DEBUG: 开始检测，self.detector={self.detector}")
        # 检查模型是否已初始化且加载成功
        if not self.detector or not hasattr(self, 'model_initialized') or not self.model_initialized:
            messagebox.showerror("错误", "模型未加载，请稍后再试")
            return
        
        input_type = self.input_type.get()
        file_path = self.file_path_var.get()
        print(f"DEBUG: 检测开始时的文件路径: {file_path}")
        
        # 检查输入
        if input_type in ["image", "video"] and not file_path:
            messagebox.showerror("错误", "请选择输入文件")
            return
        
        # 规范化文件路径并检查是否存在
        if input_type in ["image", "video"]:
            normalized_path = os.path.normpath(file_path)
            print(f"DEBUG: 规范化后的文件路径: {normalized_path}")
            
            # 确保路径编码正确（特别是包含中文的路径）
            try:
                # 使用UTF-8编码处理路径
                if isinstance(normalized_path, str):
                    # 确保路径字符串使用正确的编码
                    normalized_path = normalized_path.encode('utf-8').decode('utf-8')
            except Exception as e:
                print(f"DEBUG: 路径编码处理异常: {e}")
            
            if not os.path.exists(normalized_path):
                messagebox.showerror("错误", f"文件不存在: {normalized_path}")
                return
            # 更新为规范化后的路径
            file_path = normalized_path
        
        # 更新模型参数
        self.detector.confidence_threshold = self.confidence_var.get()
        self.detector.nms_threshold = self.nms_var.get()
        
        # 根据输入类型启动相应的检测
        if input_type == "image":
            self.is_processing = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.progress_frame.pack(fill=tk.X, pady=(10, 0))
            self.progress_var.set(0)
            self.progress_label.config(text="正在处理图片...")
            
            # 在新线程中执行检测
            self.processing_thread = threading.Thread(target=self.process_image, args=(file_path,))
            self.processing_thread.daemon = True
            self.processing_thread.start()
        elif input_type == "video":
            self.is_processing = True
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.progress_frame.pack(fill=tk.X, pady=(10, 0))
            self.progress_var.set(0)
            self.progress_label.config(text="正在处理视频...")
            
            # 在新线程中执行检测
            self.processing_thread = threading.Thread(target=self.process_video, args=(file_path,))
            self.processing_thread.daemon = True
            self.processing_thread.start()
        elif input_type == "camera":
            self.start_camera_detection()
    
    def stop_detection(self):
        """停止检测"""
        self.is_processing = False
        self.is_camera_active = False
        self.stop_button.config(state='disabled')
        self.progress_label.config(text="正在停止...")
    
    def finish_detection(self):
        """完成检测"""
        self.is_processing = False
        self.is_camera_active = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
    
    def detection_error(self, error_message):
        """检测错误处理"""
        messagebox.showerror("检测错误", error_message)
        self.status_var.set(f"错误: {error_message}")
        self.finish_detection()
    
    def on_confidence_change(self, value):
        """置信度阈值变化回调"""
        # 更新标签显示
        self.confidence_label.config(text=f"{float(value):.2f}")
        
    def on_nms_change(self, value):
        """NMS阈值变化回调"""
        # 更新标签显示
        self.nms_label.config(text=f"{float(value):.2f}")
    
    def on_input_type_change(self):
        """输入类型变化回调"""
        input_type = self.input_type.get()
        if input_type in ["image", "video"]:
            # 显示文件选择区域，隐藏摄像头设置
            self.file_frame.pack(fill=tk.X, pady=(0, 10))
            self.camera_frame.pack_forget()
        elif input_type == "camera":
            # 显示摄像头设置，隐藏文件选择区域
            self.camera_frame.pack(fill=tk.X, pady=(0, 10))
            self.file_frame.pack_forget()
    
    def on_mousewheel(self, event):
        """鼠标滚轮事件处理"""
        self.image_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def on_closing(self):
        """窗口关闭事件"""
        self.is_processing = False
        self.is_camera_active = False
        
        # 释放摄像头资源
        if self.video_capture:
            self.video_capture.release()
        
        # 等待线程结束
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)
        
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1)
        
        self.root.destroy()

def main():
    root = tk.Tk()
    app = YOLOGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()