import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time
from matplotlib.patches import Rectangle
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, BOTH, YES
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import queue
import PIL.Image
import PIL.ImageTk

# 全局作用域定义ZoomableCanvas类，确保在使用前已定义
class ZoomableCanvas:
    """
    支持鼠标滚轮缩放和拖动的画布类
    """
    def __init__(self, figure, master):
        self.figure = figure
        self.master = master  # 保存父容器引用
        self.canvas = FigureCanvasTkAgg(figure, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=BOTH, expand=YES)
        
        # 缩放和拖动参数
        self.xdata = 0
        self.ydata = 0
        self.scale = 1.0
        self.press = None
        
        # 绑定事件
        self.canvas_widget.bind("<Button-1>", self.on_press)
        self.canvas_widget.bind("<B1-Motion>", self.on_motion)
        self.canvas_widget.bind("<ButtonRelease-1>", self.on_release)
        self.canvas_widget.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.canvas_widget.bind("<Button-4>", self.on_mousewheel_linux)  # Linux
        self.canvas_widget.bind("<Button-5>", self.on_mousewheel_linux)  # Linux
        
        # 监听窗口大小变化
        master.bind("<Configure>", self.on_master_configure)
    
    def on_press(self, event):
        """鼠标按下事件"""
        self.press = event.x, event.y
    
    def on_motion(self, event):
        """鼠标拖动事件"""
        if self.press is None:
            return
        x, y = self.press
        dx = event.x - x
        dy = event.y - y
        self.press = event.x, event.y
        
        # 更新视图位置
        for ax in self.figure.axes:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # 计算缩放因子对应的实际移动距离
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            dx_data = -(dx / self.canvas_widget.winfo_width()) * x_range
            dy_data = (dy / self.canvas_widget.winfo_height()) * y_range
            
            ax.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
            ax.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
        
        self.canvas.draw_idle()
    
    def on_release(self, event):
        """鼠标释放事件"""
        self.press = None
    
    def on_mousewheel(self, event):
        """Windows鼠标滚轮事件"""
        # 确定缩放方向 - 已调换方向，向下滚动放大，向上滚动缩小
        scale_factor = 0.9 if event.delta > 0 else 1.1
        self._scale_canvas(event.x, event.y, scale_factor)
        # 阻止事件冒泡，避免触发外层画布的滚动
        return "break"
    
    def on_mousewheel_linux(self, event):
        """Linux鼠标滚轮事件"""
        # 确定缩放方向 - 已调换方向，向下滚动放大，向上滚动缩小
        scale_factor = 0.9 if event.num == 4 else 1.1
        self._scale_canvas(event.x, event.y, scale_factor)
        # 阻止事件冒泡，避免触发外层画布的滚动
        return "break"
    
    def _scale_canvas(self, x, y, scale_factor):
        """缩放画布"""
        for ax in self.figure.axes:
            # 获取当前鼠标位置对应的坐标
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # 计算鼠标在轴上的相对位置
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            # 计算缩放中心点
            x_center = xlim[0] + (x / self.canvas_widget.winfo_width()) * x_range
            y_center = ylim[0] + (y / self.canvas_widget.winfo_height()) * y_range
            
            # 应用缩放
            new_x_range = x_range * scale_factor
            new_y_range = y_range * scale_factor
            
            ax.set_xlim(
                x_center - (x_center - xlim[0]) * scale_factor,
                x_center + (xlim[1] - x_center) * scale_factor
            )
            ax.set_ylim(
                y_center - (y_center - ylim[0]) * scale_factor,
                y_center + (ylim[1] - y_center) * scale_factor
            )
        
        self.canvas.draw_idle()
    
    def destroy(self):
        """清理资源"""
        try:
            self.canvas_widget.destroy()
        except:
            pass  # 忽略已销毁组件的错误

    def on_master_configure(self, event):
        """父容器大小变化时调整图表"""
        # 避免初始调用时的错误
        if event.widget == self.master:
            # 调整画布大小以适应容器
            self.canvas_widget.configure(width=event.width - 2, height=event.height - 2)
            # 更新图表布局
            self.figure.tight_layout()
            self.canvas.draw_idle()
    
    def draw(self):
        """绘制图表"""
        # 确保图表适应容器大小
        self.figure.tight_layout()
        self.canvas.draw()


def load_yolo_model(model_name='yolov5s.pt', status_queue=None):
    """加载YOLOv5预训练模型"""
    def update_status(message):
        print(message)
        if status_queue:
            status_queue.put(f"status:{message}")

    update_status(f"正在加载YOLO模型: {model_name}")
    try:
        # 从Ultralytics加载预训练模型，添加信任仓库参数
        model = torch.hub.load(
            'ultralytics/yolov5',
            'yolov5s',
            pretrained=True,
            trust_repo=True,  # 避免安全提示
            force_reload=False  # 不重复下载模型
        )

        # 设置模型参数
        model.conf = 0.5  # 置信度阈值，只显示置信度>0.5的目标
        model.iou = 0.45  # IOU阈值，用于非极大值抑制
        update_status("YOLO模型加载完成")
        return model
    except Exception as e:
        update_status(f"模型加载失败: {str(e)}")
        raise


def detect_objects(model, image_path, status_queue=None, progress_queue=None):
    """使用YOLO模型检测图片中的目标"""
    def update_status(message):
        print(message)
        if status_queue:
            status_queue.put(f"status:{message}")

    def update_progress(value):
        if progress_queue:
            progress_queue.put(value)

    update_status(f"正在处理图片: {os.path.basename(image_path)}")
    update_progress(10)
    start_time = time.time()

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}，请检查路径是否正确")
    update_progress(20)

    # 记录原始图像信息
    original_h, original_w = img.shape[:2]
    update_status(f"图片原始尺寸: {original_w}x{original_h}")

    # 缩放大图片，避免内存占用过高
    max_size = 1024  # 最大边长限制
    if max(original_h, original_w) > max_size:
        scale = max_size / max(original_h, original_w)
        img = cv2.resize(img, (int(original_w * scale), int(original_h * scale)))
        update_status(f"图片已缩放至: {img.shape[1]}x{img.shape[0]} (缩放因子: {scale:.2f})")
    update_progress(40)

    # 转换为RGB格式（OpenCV默认是BGR）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    update_progress(60)

    # 进行目标检测
    update_status("正在运行目标检测...")
    update_progress(70)
    results = model(img_rgb)
    update_progress(90)
    detection_time = time.time() - start_time
    update_status(f"检测完成，耗时: {detection_time:.2f} 秒")
    update_progress(100)

    return img_rgb, results, detection_time


def create_visualization_figures(img, results, detection_time, image_path):
    """
    创建多个独立的可视化图表，每个图表作为单独的figure对象
    
    返回:
        figures: 包含9个独立图表的字典
        detections: 检测结果数据框
        total_objects: 检测到的目标总数
    """
    # 确保matplotlib在Tk环境中正常工作
    import matplotlib
    matplotlib.use('TkAgg')  # 确保使用TkAgg后端
    import matplotlib.pyplot as plt
    
    # 配置matplotlib字体，确保中文正常显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    
    print("===== 调试: 进入create_visualization_figures函数 =====")
    print(f"调试: results类型: {type(results)}")
    print(f"调试: results属性: {dir(results)}")
    
    # 创建结果字典
    figures = {}
    
    # 处理检测结果
    detections = []
    
    # 确保detections是列表
    if not isinstance(detections, list):
        detections = []
    
    # 尝试多种方式获取检测框数据
    try:
        # 方式1: 直接访问boxes属性
        if hasattr(results, 'boxes'):
            print("调试: 使用results.boxes")
            boxes = results.boxes
            for box in boxes:
                if hasattr(box, 'cls') and len(box.cls) > 0:
                    class_id = int(box.cls[0])
                    class_name = results.names[class_id] if hasattr(results, 'names') else f"类{class_id}"
                    confidence = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0
                    if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                        detections.append({
                            'name': class_name,
                            'confidence': confidence,
                            'xmin': xmin,
                            'ymin': ymin,
                            'xmax': xmax,
                            'ymax': ymax
                        })
        # 方式2: 检查是否有pred属性(YOLOv5输出格式)
        elif hasattr(results, 'pred'):
            print("调试: 使用results.pred")
            for pred in results.pred:
                for *box, conf, cls in pred:
                    class_id = int(cls)
                    class_name = results.names[class_id] if hasattr(results, 'names') else f"类{class_id}"
                    xmin, ymin, xmax, ymax = map(int, box)
                    detections.append({
                        'name': class_name,
                        'confidence': float(conf),
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax
                    })
        # 方式3: 检查是否是列表格式
        elif isinstance(results, list) and len(results) > 0:
            print("调试: results是列表格式")
            for result in results:
                if hasattr(result, 'boxes'):
                    for box in result.boxes:
                        if hasattr(box, 'cls') and len(box.cls) > 0:
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id] if hasattr(result, 'names') else f"类{class_id}"
                            confidence = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0
                            if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                                detections.append({
                                    'name': class_name,
                                    'confidence': confidence,
                                    'xmin': xmin,
                                    'ymin': ymin,
                                    'xmax': xmax,
                                    'ymax': ymax
                                })
        else:
            print(f"警告: 无法识别results对象的格式，类型: {type(results)}")
    except Exception as e:
        print(f"错误: 处理检测结果时出错: {e}")
    
    # 转换为DataFrame
    import pandas as pd
    detections_df = pd.DataFrame(detections)
    total_objects = len(detections_df)
    print(f"调试: 检测到的目标总数: {total_objects}")
    
    # 1. 原始图像
    fig1 = plt.figure(figsize=(8, 6), dpi=100)
    fig1.suptitle("原始图像", fontsize=14)
    ax1 = fig1.add_subplot(111)
    ax1.imshow(img)
    ax1.axis('off')
    # 不在这里反转y轴，因为matplotlib.imshow默认会正确显示图像
    figures['original'] = fig1
    
    # 2. 检测结果图像
    fig2 = plt.figure(figsize=(8, 6), dpi=100)
    fig2.suptitle("检测结果", fontsize=14)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(img)
    ax2.axis('off')
    # 不在这里反转y轴，因为matplotlib.imshow默认会正确显示图像
    
    # 绘制边界框和标签
    # 获取图像高度以调整y坐标
    img_height = img.shape[0] if hasattr(img, 'shape') else 0
    
    for idx, row in detections_df.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        class_name = row['name']
        confidence = row['confidence']
        
        # 因为我们不再反转y轴，需要调整y坐标
        if img_height > 0:
            # 转换y坐标，使其在不反转y轴的情况下正确显示
            ymin_adj = img_height - ymax
            ymax_adj = img_height - ymin
            ymin, ymax = ymin_adj, ymax_adj
        
        # 绘制边界框
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                           fill=False, edgecolor='red', linewidth=2)
        ax2.add_patch(rect)
        
        # 添加标签
        label = f'{class_name} ({confidence:.2f})'
        ax2.text(xmin, ymin - 5, label, color='white', fontsize=10, 
                bbox=dict(facecolor='red', alpha=0.7, boxstyle='round,pad=0.5'))
    figures['detection'] = fig2
    
    # 3. 检测统计摘要
    fig3 = plt.figure(figsize=(8, 6), dpi=100)
    fig3.suptitle("检测统计摘要", fontsize=14)
    ax3 = fig3.add_subplot(111)
    
    # 无检测目标时的处理
    if total_objects == 0:
        ax3.text(0.5, 0.5, "未检测到任何目标", ha='center', va='center', fontsize=14)
        ax3.axis('off')
    else:
        # 计算统计信息
        class_counts = detections_df['name'].value_counts()
        avg_confidence = detections_df['confidence'].mean()
        
        # 添加统计文本
        stats_text = f"总检测目标数: {total_objects}\n"
        stats_text += f"平均置信度: {avg_confidence:.2f}\n\n"
        stats_text += "各类别数量:\n"
        for cls, count in class_counts.items():
            stats_text += f"{cls}: {count}个\n"
        
        ax3.text(0.1, 0.9, stats_text, ha='left', va='top', fontsize=12, family='monospace')
        ax3.axis('off')
    figures['stats'] = fig3
    
    # 4. 目标类别分布饼图
    fig4 = plt.figure(figsize=(8, 6), dpi=100)
    fig4.suptitle("目标类别分布饼图", fontsize=14)
    ax4 = fig4.add_subplot(111)
    
    if total_objects > 0:
        class_counts = detections_df['name'].value_counts()
        ax4.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
               startangle=90, shadow=True, explode=[0.05] * len(class_counts))
        ax4.axis('equal')  # 保持饼图为圆形
    else:
        ax4.text(0.5, 0.5, "未检测到任何目标", ha='center', va='center', fontsize=14)
        ax4.axis('off')
    figures['pie_chart'] = fig4
    
    # 5. 目标置信度分布直方图
    fig5 = plt.figure(figsize=(8, 6), dpi=100)
    fig5.suptitle("目标置信度分布直方图", fontsize=14)
    ax5 = fig5.add_subplot(111)
    
    if total_objects > 0:
        ax5.hist(detections_df['confidence'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_xlabel('置信度')
        ax5.set_ylabel('数量')
        ax5.grid(True, linestyle='--', alpha=0.7)
    else:
        ax5.text(0.5, 0.5, "未检测到任何目标", ha='center', va='center', fontsize=14)
        ax5.axis('off')
    figures['confidence_hist'] = fig5
    
    # 6. 目标大小分布直方图
    fig6 = plt.figure(figsize=(8, 6), dpi=100)
    fig6.suptitle("目标大小分布直方图", fontsize=14)
    ax6 = fig6.add_subplot(111)
    
    if total_objects > 0:
        # 计算目标面积
        detections_df['area'] = (detections_df['xmax'] - detections_df['xmin']) * \
                               (detections_df['ymax'] - detections_df['ymin'])
        ax6.hist(detections_df['area'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        ax6.set_xlabel('目标大小 (像素²)')
        ax6.set_ylabel('数量')
        ax6.grid(True, linestyle='--', alpha=0.7)
    else:
        ax6.text(0.5, 0.5, "未检测到任何目标", ha='center', va='center', fontsize=14)
        ax6.axis('off')
    figures['size_hist'] = fig6
    
    # 7. 目标空间分布热力图
    fig7 = plt.figure(figsize=(8, 6), dpi=100)
    fig7.suptitle("目标空间分布热力图", fontsize=14)
    ax7 = fig7.add_subplot(111)
    
    if total_objects > 0:
        # 创建热力图数据
        heatmap_data = np.zeros((img.shape[0] // 10, img.shape[1] // 10))
        for _, row in detections_df.iterrows():
            center_x = (row['xmin'] + row['xmax']) // 2
            center_y = (row['ymin'] + row['ymax']) // 2
            h_x, h_y = center_x // 10, center_y // 10
            if 0 <= h_y < heatmap_data.shape[0] and 0 <= h_x < heatmap_data.shape[1]:
                heatmap_data[h_y, h_x] += 1
        
        # 显示热力图
        im = ax7.imshow(heatmap_data, cmap='hot', interpolation='nearest')
        fig7.colorbar(im, ax=ax7, label='目标密度')
        ax7.set_xlabel('图像宽度 (缩放)')
        ax7.set_ylabel('图像高度 (缩放)')
    else:
        ax7.text(0.5, 0.5, "未检测到任何目标", ha='center', va='center', fontsize=14)
        ax7.axis('off')
    figures['heatmap'] = fig7
    
    # 8. 类别置信度对比条形图
    fig8 = plt.figure(figsize=(8, 6), dpi=100)
    fig8.suptitle("类别置信度对比条形图", fontsize=14)
    ax8 = fig8.add_subplot(111)
    
    if total_objects > 0:
        # 按类别计算平均置信度
        class_confidence = detections_df.groupby('name')['confidence'].mean().sort_values(ascending=False)
        bars = ax8.bar(class_confidence.index, class_confidence.values, alpha=0.7, color='orange')
        ax8.set_xlabel('类别')
        ax8.set_ylabel('平均置信度')
        ax8.set_ylim(0, 1)
        ax8.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45, ha='right')
    else:
        ax8.text(0.5, 0.5, "未检测到任何目标", ha='center', va='center', fontsize=14)
        ax8.axis('off')
    figures['class_confidence'] = fig8
    
    # 9. 性能指标表格
    fig9 = plt.figure(figsize=(8, 6), dpi=100)
    fig9.suptitle("检测任务性能指标", fontsize=14)
    ax9 = fig9.add_subplot(111)
    
    # 创建性能指标数据
    performance_data = [
        ['总检测目标数', str(total_objects)],
        ['检测耗时', f'{detection_time:.2f} 秒'],
        ['平均处理速度', f'{1/detection_time:.2f} 帧/秒' if detection_time > 0 else 'N/A']
    ]
    
    if total_objects > 0:
        # 添加更多统计信息
        performance_data.extend([
            ['平均置信度', f'{detections_df["confidence"].mean():.2f}'],
            ['检测类别数', str(detections_df['name'].nunique())]
        ])
    
    # 创建表格
    table = ax9.table(cellText=performance_data, colLabels=['指标', '值'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # 美化表格
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 表头
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color='white')
        else:  # 数据行
            if i % 2 == 0:
                cell.set_facecolor('#f5f5f5')
    
    ax9.axis('off')  # 隐藏坐标轴
    figures['performance'] = fig9
    
    print(f"===== 调试: 函数结束，返回图表数量: {len(figures)} =====")
    return figures, detections_df, total_objects

# 为了向后兼容，保留原有的函数名
create_visualization_figure = create_visualization_figures

def save_results(img_path, detections, total_objects):
    """保存检测结果为文件"""
    # 保存结果图片
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成保存文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # 保存检测数据为CSV文件
    csv_path = os.path.join(output_dir, f"{base_name}_detections_{timestamp}.csv")
    
    if total_objects > 0:
        # 增强CSV数据，添加额外信息
        enhanced_detections = detections.copy()
        # 计算目标面积
        enhanced_detections['area'] = (enhanced_detections['xmax'] - enhanced_detections['xmin']) * \
                                     (enhanced_detections['ymax'] - enhanced_detections['ymin'])
        # 计算目标中心点
        enhanced_detections['center_x'] = (enhanced_detections['xmin'] + enhanced_detections['xmax']) / 2
        enhanced_detections['center_y'] = (enhanced_detections['ymin'] + enhanced_detections['ymax']) / 2
        # 计算目标宽高比
        enhanced_detections['aspect_ratio'] = (enhanced_detections['xmax'] - enhanced_detections['xmin']) / \
                                            (enhanced_detections['ymax'] - enhanced_detections['ymin'])
        # 替换NaN值
        enhanced_detections = enhanced_detections.fillna('N/A')
        
        # 保存增强后的数据
        enhanced_detections.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"增强版检测数据已保存至: {csv_path}")
        
        # 同时保存一份汇总统计报告
        stats_report_path = os.path.join(output_dir, f"{base_name}_summary_{timestamp}.txt")
        with open(stats_report_path, 'w', encoding='utf-8') as f:
            f.write(f"YOLOv5 目标检测统计报告\n")
            f.write(f"================================\n")
            f.write(f"图片名称: {os.path.basename(img_path)}\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总检测到目标数: {total_objects}\n\n")
            
            # 类别统计
            class_counts = enhanced_detections['name'].value_counts()
            f.write("类别统计:\n")
            for cls, count in class_counts.items():
                f.write(f"  - {cls}: {count}个\n")
            f.write("\n")
            
            # 置信度统计
            f.write("置信度统计:\n")
            f.write(f"  - 平均值: {enhanced_detections['confidence'].mean():.3f}\n")
            f.write(f"  - 最大值: {enhanced_detections['confidence'].max():.3f}\n")
            f.write(f"  - 最小值: {enhanced_detections['confidence'].min():.3f}\n")
            f.write(f"  - 标准差: {enhanced_detections['confidence'].std():.3f}\n\n")
            
            # 尺寸统计
            f.write("目标尺寸统计:\n")
            f.write(f"  - 平均面积: {enhanced_detections['area'].mean():.1f} 像素²\n")
            f.write(f"  - 最大面积: {enhanced_detections['area'].max():.1f} 像素²\n")
            f.write(f"  - 最小面积: {enhanced_detections['area'].min():.1f} 像素²\n\n")
            
            # 位置分布
            f.write("位置分布统计:\n")
            f.write(f"  - 中心点X范围: {enhanced_detections['center_x'].min():.1f} - {enhanced_detections['center_x'].max():.1f}\n")
            f.write(f"  - 中心点Y范围: {enhanced_detections['center_y'].min():.1f} - {enhanced_detections['center_y'].max():.1f}\n")
        
        print(f"检测统计报告已保存至: {stats_report_path}")
        return csv_path
    return None

class YOLODetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5 目标检测可视化工具")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 800)
        
        # 设置中文字体，只使用最常用的SimHei字体以避免字体查找错误
        plt.rcParams["font.family"] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False
        
        # 变量初始化
        self.model = None
        self.image_path = None
        self.detection_thread = None
        self.status_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.is_processing = False
        
        # 创建UI
        self.create_widgets()
        
        # 启动模型加载线程
        self.load_model_thread()
        
        # 启动消息处理循环
        self.process_messages()
        
        # 绑定窗口关闭事件处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def on_closing(self):
        """窗口关闭事件处理"""
        print("程序正在关闭...")
        
        # 设置退出标志
        self._should_quit = True
        
        # 停止正在运行的线程
        self.is_processing = False
        
        # 停止消息处理循环
        if hasattr(self, '_message_job_id'):
            try:
                self.root.after_cancel(self._message_job_id)
            except:
                pass
        
        # 清理所有图表资源
        try:
            # 清理所有图表画布
            if hasattr(self, 'chart_canvases'):
                for canvas_info in self.chart_canvases:
                    try:
                        if 'canvas' in canvas_info and canvas_info['canvas']:
                            canvas_info['canvas'].get_tk_widget().destroy()
                    except:
                        pass
        except:
            pass
        
        # 关闭主窗口
        self.root.destroy()
        print("程序已正确关闭")
    
    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # 图片选择按钮
        self.select_button = ttk.Button(control_frame, text="选择图片", command=self.select_image)
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        # 图片路径显示
        self.image_path_var = tk.StringVar()
        self.image_path_entry = ttk.Entry(control_frame, textvariable=self.image_path_var, width=70)
        self.image_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 运行检测按钮
        self.run_button = ttk.Button(control_frame, text="运行检测", command=self.start_detection, state=tk.DISABLED)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        # 测试图表显示按钮
        self.test_button = ttk.Button(control_frame, text="测试图表显示", command=self.test_visualization)
        self.test_button.pack(side=tk.LEFT, padx=5)
        
        # 状态和进度区域
        status_frame = ttk.LabelFrame(main_frame, text="处理状态", padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, length=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # 状态文本
        self.status_var = tk.StringVar()
        self.status_var.set("准备就绪，正在加载模型...")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, wraplength=1000)
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # 结果统计区域
        self.stats_frame = ttk.LabelFrame(main_frame, text="检测统计", padding="10")
        self.stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_var = tk.StringVar()
        self.stats_var.set("请选择图片并运行检测")
        self.stats_label = ttk.Label(self.stats_frame, textvariable=self.stats_var, wraplength=1000)
        self.stats_label.pack(fill=tk.X, padx=5, pady=5)
        
        # 可视化结果区域
        self.visualization_frame = ttk.LabelFrame(main_frame, text="可视化结果", padding="10")
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建一个占位标签
        self.placeholder_label = ttk.Label(self.visualization_frame, text="检测结果将显示在这里")
        self.placeholder_label.pack(expand=True)
        
        # 初始化画布和工具栏
        self.canvas = None
        self.toolbar = None
    
    def load_model_thread(self):
        """在单独线程中加载模型"""
        self.detection_thread = threading.Thread(target=self._load_model_task)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def test_visualization(self):
        """测试图表显示功能"""
        print("===== 开始测试图表显示 =====")
        
        # 清空结果区域
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        try:
            # 创建一个简单的折线图
            import numpy as np
            fig = plt.Figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # 生成示例数据
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            
            # 绘制图表
            ax.plot(x, y, 'b-', label='正弦曲线')
            ax.set_title('测试图表显示', fontsize=14)
            ax.set_xlabel('X轴', fontsize=12)
            ax.set_ylabel('Y轴', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # 使用FigureCanvasTkAgg显示图表
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 绘制图表
            canvas.draw()
            
            print("测试图表显示成功")
            
        except Exception as e:
            print(f"测试图表显示失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 显示错误信息
            error_label = ttk.Label(self.result_frame, text=f"测试失败: {str(e)}", font=('SimHei', 12), foreground='red')
            error_label.pack(padx=20, pady=20)
    
    def _load_model_task(self):
        """模型加载任务"""
        try:
            self.model = load_yolo_model(status_queue=self.status_queue)
            self.status_queue.put("model_loaded")
        except Exception as e:
            self.status_queue.put(f"error:{str(e)}")
    
    def select_image(self):
        """选择图片文件"""
        file_types = [
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("所有文件", "*.*")
        ]
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=file_types)

        if file_path:
            self.image_path = file_path
            self.image_path_var.set(file_path)

            # 如果模型已加载，启用运行按钮
            if self.model is not None:
                self.run_button.config(state=tk.NORMAL)

            # 清除之前的结果
            self.clear_results()
    
    def start_detection(self):
        """开始目标检测"""
        if not self.image_path or not os.path.exists(self.image_path):
            messagebox.showerror("错误", "请先选择有效的图片文件")
            return
        
        if self.is_processing:
            messagebox.showinfo("提示", "检测正在进行中，请稍候...")
            return
        
        self.is_processing = True
        self.run_button.config(state=tk.DISABLED)
        self.select_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        # 启动检测线程
        self.detection_thread = threading.Thread(target=self._detection_task)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def _detection_task(self):
        """检测任务"""
        try:
            # 检测目标
            img, results, detection_time = detect_objects(
                self.model, 
                self.image_path, 
                status_queue=self.status_queue,
                progress_queue=self.progress_queue
            )

            # 保存中间结果到实例变量，避免在队列中传递复杂对象
            self.last_results = {
                'img': img,
                'results': results,
                'detection_time': detection_time,
                'image_path': self.image_path
            }

            # 通知UI线程检测完成
            self.status_queue.put("detection_complete")
        except Exception as e:
            self.status_queue.put(f"error:{str(e)}")
    
    def clear_results(self):
        """清除结果显示"""
        # 清除之前的画布和工具栏
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            self.canvas = None

        if self.toolbar:
            self.toolbar.pack_forget()
            self.toolbar = None
        
        # 显示占位符
        self.placeholder_label.pack(expand=True)
        
        # 清除统计信息
        self.stats_var.set("请选择图片并运行检测")
    
    def update_visualization(self, fig, detections, total_objects, csv_path):
        """
        更新可视化结果，显示多个独立的图表，每个图表都支持鼠标滚轮缩放和拖动
        """
        # 清除之前的可视化内容
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()
        
        # 创建一个可滚动的框架来容纳多个图表
        scrollable_frame = ttk.Frame(self.visualization_frame)
        scrollable_frame.pack(fill=tk.BOTH, expand=tk.YES)
        
        # 创建滚动条
        tk_canvas = tk.Canvas(scrollable_frame)
        scrollbar = ttk.Scrollbar(scrollable_frame, orient="vertical", command=tk_canvas.yview)
        scrollable_content = ttk.Frame(tk_canvas)
        
        scrollable_content.bind(
            "<Configure>",
            lambda e: tk_canvas.config(
                scrollregion=tk_canvas.bbox("all")
            )
        )
        
        tk_canvas.create_window((0, 0), window=scrollable_content, anchor="nw")
        tk_canvas.config(yscrollcommand=scrollbar.set)     
        
        # 显示滚动条和画布
        scrollbar.pack(side="right", fill="y")
        tk_canvas.pack(side="left", fill=tk.BOTH, expand=tk.YES)
        
        # 存储引用以便后续访问
        self.scrollable_frame = scrollable_frame
        self.scrollable_canvas = tk_canvas
        self.scrollable_content = scrollable_content
        
        # 添加改进的鼠标滚轮滚动支持
        def on_canvas_mousewheel(event):
            try:
                # 获取鼠标下方的组件
                widget_under_mouse = scrollable_content.winfo_containing(event.x_root, event.y_root)
                
                # 检查是否在图像图表区域
                is_in_image_chart = False
                current_widget = widget_under_mouse
                
                # 向上查找是否在图像图表容器中
                while current_widget and current_widget != scrollable_content:
                    if isinstance(current_widget, ttk.LabelFrame):
                        # 检查标题是否是图像图表
                        if current_widget.cget("text") in ["原始图像", "检测结果"]:
                            is_in_image_chart = True
                            break
                    current_widget = current_widget.master
                
                # 只有当不在图像图表区域时才滚动大画布
                if not is_in_image_chart:
                    if event.delta > 0:
                        # 使用正确的滚动画布对象
                        self.scrollable_canvas.yview_scroll(-1, "units")
                    else:
                        self.scrollable_canvas.yview_scroll(1, "units")
            except Exception:
                # 如果出错，默认执行滚动
                try:
                    if event.delta > 0:
                        self.scrollable_canvas.yview_scroll(-1, "units")
                    else:
                        self.scrollable_canvas.yview_scroll(1, "units")
                except:
                    pass
        
        # 为画布和滚动框架都绑定滚轮事件，确保任何地方都能滚动
        tk_canvas.bind("<MouseWheel>", on_canvas_mousewheel)
        scrollable_frame.bind("<MouseWheel>", on_canvas_mousewheel)
        
        # 尝试使用新的多图表函数
        try:
            print("===== 调试: 开始update_visualization方法 =====")
            # 检查last_results是否存在且包含必要的数据
            if not hasattr(self, 'last_results') or not self.last_results:
                print("错误: self.last_results不存在或为空")
                # 创建一个错误提示标签
                error_label = ttk.Label(scrollable_content, text="错误: 没有检测结果数据", font=('SimHei', 12))
                error_label.pack(padx=20, pady=20)
                return
            
            # 检查必要的键是否存在
            required_keys = ['img', 'results', 'detection_time', 'image_path']
            for key in required_keys:
                if key not in self.last_results:
                    print(f"错误: self.last_results中缺少{key}键")
                    error_label = ttk.Label(scrollable_content, text=f"错误: 缺少必要的检测结果数据", font=('SimHei', 12))
                    error_label.pack(padx=20, pady=20)
                    return
            
            # 直接调用函数获取所有图表
            print("调试: 尝试创建多图表")
            try:
                # 捕获create_visualization_figures可能的错误
                result = create_visualization_figures(self.last_results['img'], 
                                                    self.last_results['results'], 
                                                    self.last_results['detection_time'], 
                                                    self.last_results['image_path'])
                
                # 检查返回值
                if len(result) < 3:
                    print("错误: create_visualization_figures返回值不完整")
                    error_label = ttk.Label(scrollable_content, text="错误: 图表创建失败", font=('SimHei', 12))
                    error_label.pack(padx=20, pady=20)
                    return
                
                figures, detections_df, total_objects = result
                
                # 验证figures是字典且不为空
                if not isinstance(figures, dict) or not figures:
                    print("错误: figures不是有效的字典或为空")
                    error_label = ttk.Label(scrollable_content, text="错误: 没有创建有效的图表", font=('SimHei', 12))
                    error_label.pack(padx=20, pady=20)
                    return
                
                print(f"调试: 成功创建图表数量: {len(figures)}")
            except Exception as e:
                print(f"错误: 创建图表失败: {str(e)}")
                error_label = ttk.Label(scrollable_content, text=f"图表创建失败: {str(e)}", font=('SimHei', 10), foreground='red')
                error_label.pack(padx=10, pady=10)
                return
            
            # 创建两列布局的框架
            columns = []
            for i in range(2):
                col_frame = ttk.Frame(scrollable_content)
                col_frame.pack(side="left", fill=tk.BOTH, expand=tk.YES, padx=5)
                columns.append(col_frame)
                print(f"调试: 创建列框架 {i}")
            
            # 存储所有图表信息，用于后续更新
            self.chart_info = []
            
            # 显示每个图表，实现两列布局
            for i, (name, figure) in enumerate(figures.items()):
                try:
                    print(f"调试: 添加图表 {name} 到界面")
                    # 确定图表应该放在哪一列
                    col_index = i % 2
                    
                    # 获取图表标题
                    chart_title = self._get_figure_title(name)
                    
                    # 为前两个图表（原始图像和检测结果）取消边框，使用普通Frame
                    if i < 2:  # 第一个和第二个图表
                        chart_container = ttk.Frame(columns[col_index])
                        # 添加标题标签
                        title_label = ttk.Label(chart_container, text=chart_title, font=('SimHei', 12, 'bold'))
                        title_label.pack(side='top', fill=tk.X, pady=(0, 5))
                    else:  # 其他图表保持原有样式
                        chart_container = ttk.LabelFrame(columns[col_index], text=chart_title)
                    
                    # 设置填充和扩展属性，确保图表能根据窗口大小调整
                    chart_container.pack(fill=tk.BOTH, expand=tk.YES, padx=5, pady=5)
                    
                    # 设置图表容器的最小大小
                    chart_container.configure(width=400, height=300)
                    
                    # 使用FigureCanvasTkAgg
                    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                    canvas = FigureCanvasTkAgg(figure, master=chart_container)
                    canvas_widget = canvas.get_tk_widget()
                    canvas_widget.pack(fill=tk.BOTH, expand=tk.YES)
                    
                    # 对于第一个和第二个图表，实现缩放拖动功能
                    if i < 2:
                        # 保存当前轴的xlim和ylim
                        ax = figure.axes[0] if figure.axes else None
                        if ax:
                            # 初始设置
                            ax.set_position([0, 0, 1, 1])  # 让图表填充满整个区域
                            ax.axis('off')  # 隐藏坐标轴
                            
                            # 确保图像正确填充
                            if hasattr(ax, 'images') and ax.images:
                                img = ax.images[0]
                                # 设置图像大小匹配轴，不反转y轴
                                ax.set_xlim(0, img.get_array().shape[1])
                                ax.set_ylim(0, img.get_array().shape[0])  # 不反转y轴
                                img.set_extent([0, img.get_array().shape[1], 0, img.get_array().shape[0]])
                        
                        # 实现缩放和拖动功能 - 使用工厂函数避免闭包作用域问题
                        def create_drag_handlers(cw, c):
                            # 每个图表独立的变量
                            local_last_x, local_last_y = 0, 0
                            local_is_dragging = False
                            
                            def on_mouse_press(event):
                                nonlocal local_last_x, local_last_y, local_is_dragging
                                if event.inaxes and event.button == 1:  # 左键点击在图表区域
                                    local_last_x, local_last_y = event.x, event.y
                                    local_is_dragging = True
                            
                            def on_mouse_release(event):
                                nonlocal local_is_dragging
                                local_is_dragging = False
                            
                            def on_mouse_motion(event):
                                nonlocal local_last_x, local_last_y
                                if local_is_dragging and event.inaxes:
                                    # 计算移动距离
                                    dx = event.x - local_last_x
                                    dy = event.y - local_last_y
                                    local_last_x, local_last_y = event.x, event.y
                                    
                                    # 获取当前轴的限制
                                    xlim = event.inaxes.get_xlim()
                                    ylim = event.inaxes.get_ylim()
                                    
                                    # 计算缩放比例
                                    x_scale = (xlim[1] - xlim[0]) / cw.winfo_width()
                                    y_scale = (ylim[1] - ylim[0]) / cw.winfo_height()
                                    
                                    # 移动视图
                                    event.inaxes.set_xlim(xlim[0] - dx * x_scale, xlim[1] - dx * x_scale)
                                    event.inaxes.set_ylim(ylim[0] - dy * y_scale, ylim[1] - dy * y_scale)
                                    
                                    c.draw()
                            
                            # 返回三个处理函数
                            return on_mouse_press, on_mouse_release, on_mouse_motion
                        
                        # on_mousewheel_zoom函数已被create_zoom_handler工厂函数替代
                        
                        # 绑定事件
                        # 使用工厂函数创建并连接Matplotlib的鼠标事件处理函数
                        press_handler, release_handler, motion_handler = create_drag_handlers(canvas_widget, canvas)
                        canvas.mpl_connect('button_press_event', press_handler)
                        canvas.mpl_connect('button_release_event', release_handler)
                        canvas.mpl_connect('motion_notify_event', motion_handler)
                        
                        # 使用lambda捕获当前canvas_widget和canvas，避免闭包作用域问题
                        def create_zoom_handler(cw, c, fig):
                            def zoom_handler(event):
                                # 确保在图表区域内
                                ax = fig.axes[0] if fig.axes else None
                                if ax:
                                    # 计算鼠标在画布上的位置对应的图像坐标
                                    x, y = cw.winfo_pointerx() - cw.winfo_rootx(), \
                                           cw.winfo_pointery() - cw.winfo_rooty()
                                    
                                    # 转换为数据坐标
                                    xdata, ydata = ax.transData.inverted().transform([x, y])
                                    
                                    # 缩放因子 - 调换方向：滚轮向上缩小，向下放大
                                    scale = 0.8 if event.delta > 0 else 1.2
                                    
                                    # 获取当前轴的限制
                                    xlim = ax.get_xlim()
                                    ylim = ax.get_ylim()
                                    
                                    # 计算鼠标位置占轴的比例
                                    x_ratio = (xdata - xlim[0]) / (xlim[1] - xlim[0])
                                    y_ratio = (ydata - ylim[0]) / (ylim[1] - ylim[0])
                                    
                                    # 计算新的轴范围
                                    new_width = (xlim[1] - xlim[0]) * scale
                                    new_height = (ylim[1] - ylim[0]) * scale
                                    
                                    # 调整轴限制，保持鼠标位置相对不变
                                    new_xlim = [xdata - new_width * x_ratio, xdata + new_width * (1 - x_ratio)]
                                    new_ylim = [ydata - new_height * y_ratio, ydata + new_height * (1 - y_ratio)]
                                    
                                    # 设置新的轴限制
                                    ax.set_xlim(new_xlim)
                                    ax.set_ylim(new_ylim)
                                    
                                    # 确保图像正确填充
                                    if hasattr(ax, 'images') and ax.images:
                                        img = ax.images[0]
                                        img.set_extent([0, img.get_array().shape[1], 0, img.get_array().shape[0]])
                                    
                                    c.draw()
                            return zoom_handler
                        
                        # 为前两个图表绑定鼠标滚轮缩放事件，使用工厂函数创建独立的处理函数
                        current_zoom_handler = create_zoom_handler(canvas_widget, canvas, figure)
                        canvas_widget.bind("<MouseWheel>", current_zoom_handler)
                        
                        # Linux支持 - 使用lambda捕获当前canvas_widget
                        canvas_widget.bind("<Button-4>", lambda event, cw=canvas_widget: cw.event_generate("<MouseWheel>", delta=120))
                        canvas_widget.bind("<Button-5>", lambda event, cw=canvas_widget: cw.event_generate("<MouseWheel>", delta=-120))
                        
                        # 使用工厂函数创建并连接Matplotlib的鼠标事件处理函数
                        # 注意：这里不再使用Tkinter的bind方法，因为Tkinter事件对象和Matplotlib事件对象属性不同
                        # 已经在下方使用mpl_connect绑定了这些事件
                    else:
                        # 对于非图像图表，在X方向填充画布
                        if figure.axes:
                            ax = figure.axes[0]
                            # [left, bottom, width, height] - 调整以在X方向填充
                            ax.set_position([0.05, 0.1, 0.9, 0.85])  # X方向填充，Y方向留边距
                        
                        # 为所有图表使用相同的缩放功能，不再区分图像和非图像图表
                        # 使用lambda捕获当前canvas_widget和canvas，避免闭包作用域问题
                        def create_zoom_handler(cw, c, fig):
                            def zoom_handler(event):
                                # 确保在图表区域内
                                ax = fig.axes[0] if fig.axes else None
                                if ax:
                                    # 计算鼠标在画布上的位置对应的图像坐标
                                    x, y = cw.winfo_pointerx() - cw.winfo_rootx(), \
                                           cw.winfo_pointery() - cw.winfo_rooty()
                                    
                                    # 转换为数据坐标
                                    xdata, ydata = ax.transData.inverted().transform([x, y])
                                    
                                    # 缩放因子 - 调换方向：滚轮向上缩小，向下放大
                                    scale = 0.8 if event.delta > 0 else 1.2
                                    
                                    # 获取当前轴的限制
                                    xlim = ax.get_xlim()
                                    ylim = ax.get_ylim()
                                    
                                    # 计算鼠标位置占轴的比例
                                    x_ratio = (xdata - xlim[0]) / (xlim[1] - xlim[0]) if xlim[1] != xlim[0] else 0.5
                                    y_ratio = (ydata - ylim[0]) / (ylim[1] - ylim[0]) if ylim[1] != ylim[0] else 0.5
                                    
                                    # 计算新的轴范围
                                    new_width = (xlim[1] - xlim[0]) * scale
                                    new_height = (ylim[1] - ylim[0]) * scale
                                    
                                    # 调整轴限制，保持鼠标位置相对不变
                                    new_xlim = [xdata - new_width * x_ratio, xdata + new_width * (1 - x_ratio)]
                                    new_ylim = [ydata - new_height * y_ratio, ydata + new_height * (1 - y_ratio)]
                                    
                                    # 设置新的轴限制
                                    ax.set_xlim(new_xlim)
                                    ax.set_ylim(new_ylim)
                                    
                                    c.draw()
                            return zoom_handler
                        
                        # 为所有图表绑定鼠标滚轮缩放事件
                        current_zoom_handler = create_zoom_handler(canvas_widget, canvas, figure)
                        canvas_widget.bind("<MouseWheel>", current_zoom_handler)
                        
                        # Linux支持 - 使用lambda捕获当前canvas_widget
                        canvas_widget.bind("<Button-4>", lambda event, cw=canvas_widget: cw.event_generate("<MouseWheel>", delta=120))
                        canvas_widget.bind("<Button-5>", lambda event, cw=canvas_widget: cw.event_generate("<MouseWheel>", delta=-120))
                    
                    # 绑定窗口大小变化事件，让图表跟随缩放
                    def on_configure(event, c=canvas, f=figure, idx=i):
                        try:
                            if f.axes:
                                ax = f.axes[0]
                                # 对于所有图表，在X方向填充画布
                                # 前两个图表完全填充，其他图表保留Y方向边距以确保标签可见
                                if idx < 2:
                                    # 图像图表完全填充
                                    ax.set_position([0, 0, 1, 1])  # 让图表填充满整个区域
                                    ax.axis('off')  # 隐藏坐标轴
                                    if hasattr(ax, 'images') and ax.images:
                                        img = ax.images[0]
                                        ax.set_xlim(0, img.get_array().shape[1])
                                        ax.set_ylim(0, img.get_array().shape[0])  # 不反转y轴
                                else:
                                    # 非图像图表在X方向填充，Y方向保留边距
                                    # [left, bottom, width, height] - 调整以在X方向填充
                                    ax.set_position([0.05, 0.1, 0.9, 0.85])  # X方向填充，Y方向留边距
                            c.draw()
                        except Exception:
                            pass
                    
                    chart_container.bind("<Configure>", on_configure)
                    
                    # 立即绘制图表
                    canvas.draw()
                    
                    # 存储图表信息
                    self.chart_info.append({
                        'container': chart_container,
                        'canvas': canvas,
                        'figure': figure,
                        'canvas_widget': canvas_widget
                    })
                    
                    print(f"调试: 图表 {name} 更新完成")
                except Exception as e:
                    print(f"错误: 添加图表 {name} 失败: {str(e)}")
                    error_label = ttk.Label(scrollable_content, text=f"添加图表失败: {name}", font=('SimHei', 10), foreground='red')
                    error_label.pack(padx=10, pady=10)
                    continue
            
            print("===== 调试: update_visualization方法完成 =====")
        except Exception as e:
            # 捕获所有其他异常
            print(f"===== 调试: 出现异常: {str(e)} =====")
            import traceback
            traceback.print_exc()
            
            error_label = ttk.Label(scrollable_content, text=f"加载图表时出错: {str(e)}", font=('SimHei', 10), foreground='red')
            error_label.pack(padx=10, pady=10)
        except Exception as e:
            # 如果出现错误，显示错误信息并回退到原始的单个图表
            error_label = ttk.Label(scrollable_content, text=f"加载图表时出错: {str(e)}")
            error_label.pack(padx=10, pady=10)
            import traceback
            traceback.print_exc()
            
            # 确保创建有效的matplotlib figure对象
            import matplotlib
            if isinstance(fig, dict) or not isinstance(fig, matplotlib.figure.Figure):
                # 创建新的figure对象
                new_fig = plt.figure(figsize=(10, 8))
                ax = new_fig.add_subplot(111)
                
                # 如果有图像数据，显示图像
                if 'img' in self.last_results and self.last_results['img'] is not None:
                    ax.imshow(self.last_results['img'])
                    ax.axis('off')
                    
                    # 绘制检测框
                    if total_objects > 0:
                        for _, row in detections.iterrows():
                            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                            class_name = row['name']
                            confidence = row['confidence']
                            
                            # 获取图像高度以调整y坐标
                            img_height = self.last_results['img'].shape[0]
                            
                            # 因为我们不再反转y轴，需要调整y坐标
                            ymin_adj = img_height - ymax
                            ymax_adj = img_height - ymin
                            ymin, ymax = ymin_adj, ymax_adj
                            
                            # 绘制边界框
                            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                               fill=False, edgecolor='red', linewidth=2)
                            ax.add_patch(rect)
                            
                            # 添加标签
                            label = f'{class_name} ({confidence:.2f})'
                            ax.text(xmin, ymin - 5, label, color='white', fontsize=10, 
                                    bbox=dict(facecolor='red', alpha=0.7, boxstyle='round,pad=0.5'))
                else:
                    ax.text(0.5, 0.5, "无法显示图像", ha='center', va='center', fontsize=14)
                    ax.axis('off')
                
                # 使用创建的新figure
                display_fig = new_fig
            else:
                # fig已经是有效的figure对象，直接使用
                display_fig = fig
            
            # 创建画布显示figure
            canvas_widget = FigureCanvasTkAgg(display_fig, master=scrollable_content)
            canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)
            canvas_widget.draw()
        
        # 更新统计信息标签
        additional_stats = ""
        if total_objects > 0:
            # 计算平均置信度
            avg_confidence = detections['confidence'].mean()
            
            # 计算平均目标大小
            detections['area'] = (detections['xmax'] - detections['xmin']) * (detections['ymax'] - detections['ymin'])
            avg_size = detections['area'].mean()
            
            # 统计各个类别的数量
            class_counts = detections['name'].value_counts()
            class_info = "; ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
            
            additional_stats = f"\n平均置信度: {avg_confidence:.2f}\n" \
                              f"平均目标大小: {avg_size:.1f} 像素²\n" \
                              f"各类别数量: {class_info}"
        
        # 显示统计信息
        stats_text = f"检测统计: {total_objects} 个目标\n" \
                    f"CSV结果已保存至: {os.path.basename(csv_path)}{additional_stats}"
        
        self.stats_var.set(stats_text)
        
        # 设置统计信息标签的字体以确保中文正常显示
        self.stats_label.configure(font=('SimHei', 10))
    
    def _on_container_configure(self, container):
        """处理图表容器大小调整"""
        # 设置容器的最小大小
        container.configure(width=container.winfo_width(), height=max(300, container.winfo_height()//2))
        
        # 尝试获取容器中的ZoomableCanvas实例
        for widget in container.winfo_children():
            # 检查是否是ZoomableCanvas实例或其内部的canvas_widget
            if hasattr(widget, 'canvas'):  # 如果是ZoomableCanvas实例
                # 调整画布大小
                widget.canvas_widget.configure(width=container.winfo_width() - 20, 
                                             height=container.winfo_height() - 20)
                # 重绘画布
                widget.canvas.draw_idle()
                break
            elif isinstance(widget, tk.Canvas):  # 如果是canvas widget
                # 调整画布大小
                widget.configure(width=container.winfo_width() - 20, 
                                height=container.winfo_height() - 20)
                # 重绘画布
                widget.update_idletasks()
                break
    
    def _get_figure_title(self, name):
        """根据图表名称返回中文标题"""
        title_map = {
            'original': '1. 原始图像',
            'detection': '2. 检测结果',
            'stats': '3. 检测统计摘要',
            'pie_chart': '4. 目标类别分布饼图',
            'confidence_hist': '5. 目标置信度分布直方图',
            'size_hist': '6. 目标大小分布直方图',
            'heatmap': '7. 目标空间分布热力图',
            'class_confidence': '8. 类别置信度对比条形图',
            'performance': '9. 检测任务性能指标'
        }
        return title_map.get(name, name)
    
    def process_messages(self):
        """处理消息队列中的消息"""
        # 检查是否应该继续处理消息
        if hasattr(self, '_should_quit') and self._should_quit:
            return
            
        try:
            while not self.status_queue.empty():
                message = self.status_queue.get_nowait()
                
                if message.startswith("status:"):
                    # 更新状态文本
                    status_text = message[7:]
                    self.status_var.set(status_text)
                
                elif message == "model_loaded":
                    # 模型加载完成
                    self.status_var.set("模型加载完成，请选择图片开始检测")
                    if hasattr(self, 'image_path') and self.image_path and os.path.exists(self.image_path):
                        self.run_button.config(state=tk.NORMAL)
                
                elif message == "detection_complete":
                    # 检测完成，在UI线程中处理可视化
                    if hasattr(self, 'last_results'):
                        # 创建可视化
                        fig, detections, total_objects = create_visualization_figure(
                            self.last_results['img'], 
                            self.last_results['results'], 
                            self.last_results['detection_time'], 
                            self.last_results['image_path']
                        )
                        
                        # 保存结果
                        csv_path = save_results(
                            self.last_results['image_path'], 
                            detections, 
                            total_objects
                        )
                        
                        self.update_visualization(
                            fig, 
                            detections, 
                            total_objects, 
                            csv_path
                        )
                        
                        # 清理中间结果
                        delattr(self, 'last_results')
                    
                    self.is_processing = False
                    self.run_button.config(state=tk.NORMAL)
                    self.select_button.config(state=tk.NORMAL)
                
                elif message.startswith("error:"):
                    # 错误处理
                    error_msg = message[6:]
                    self.status_var.set(f"错误: {error_msg}")
                    messagebox.showerror("错误", error_msg)
                    self.is_processing = False
                    self.run_button.config(state=tk.NORMAL)
                    self.select_button.config(state=tk.NORMAL)
        
        except queue.Empty:
            pass
        
        # 处理进度更新
        try:
            while not self.progress_queue.empty():
                progress = self.progress_queue.get_nowait()
                self.progress_var.set(progress)
        except queue.Empty:
            pass
        
        # 继续处理消息
        self._message_job_id = self.root.after(100, self.process_messages)

def main():
    # 创建并运行GUI
    root = tk.Tk()
    app = YOLODetectorGUI(root)
    root.mainloop()

# 保留原有的命令行功能，当不通过GUI模式运行时使用
def display_results(img, results, detection_time, image_path):
    """增强的检测结果可视化展示 - 命令行模式"""
    fig, detections, total_objects = create_visualization_figure(
        img, results, detection_time, image_path
    )
    
    # 统计信息
    class_counts = detections['name'].value_counts()
    
    print(f"\n=== 检测结果统计 ===")
    print(f"总检测到目标数量: {total_objects}")
    print(f"检测耗时: {detection_time:.2f} 秒")
    print(f"\n各类别数量:")
    for cls, count in class_counts.items():
        print(f"- {cls}: {count}个")
    
    # 打印详细检测信息
    if total_objects > 0:
        print("\n详细检测信息:")
        for index, row in detections.iterrows():
            print(f"[{index+1}] 类别: {row['name']}, 置信度: {row['confidence']:.2f}, "
                  f"位置: ({int(row['xmin'])}, {int(row['ymin'])}) 到 ({int(row['xmax'])}, {int(row['ymax'])})")
    
    # 保存结果
    save_results(image_path, detections, total_objects)
    
    # 保存可视化图片
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{base_name}_results_{timestamp}.png")
    
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化结果已保存至: {save_path}")
    
    # 显示结果
    plt.show()


if __name__ == "__main__":
    # 启动GUI应用
    main()
