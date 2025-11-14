# 导出为TensorRT格式（.engine）
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="engine", imgsz=640)  # 仅需导出一次
# 加载加速模型
from ultralytics import YOLO
import sqlite3
from datetime import datetime


# 1. 初始化数据库和表
def init_db():
    # 连接数据库（不存在则创建）
    conn = sqlite3.connect('detection_results.db')
    cursor = conn.cursor()
    # 创建表：存储检测时间、文件名、目标类别、坐标、置信度
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS detection
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       timestamp
                       DATETIME,
                       #
                       检测时间
                       filename
                       TEXT,
                       #
                       文件名
                       class_name
                       TEXT,
                       #
                       目标类别
                       x1
                       REAL,
                       y1
                       REAL,
                       #
                       边界框左上角坐标
                       x2
                       REAL,
                       y2
                       REAL,
                       #
                       边界框右下角坐标
                       confidence
                       REAL
                       #
                       置信度
                   )
                   ''')
    conn.commit()
    conn.close()


# 2. 存储单条检测结果到数据库
def save_to_db(filename, class_name, x1, y1, x2, y2, confidence):
    conn = sqlite3.connect('detection_results.db')
    cursor = conn.cursor()
    # 插入数据（当前时间+检测信息）
    cursor.execute('''
                   INSERT INTO detection (timestamp, filename, class_name, x1, y1, x2, y2, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ''', (datetime.now(), filename, class_name, x1, y1, x2, y2, confidence))
    conn.commit()
    conn.close()


# 3. 模型检测并存储结果
def detect_and_save_to_db(video_path):
    init_db()  # 初始化数据库
    model = YOLO("yolov8n.pt")
    results = model(video_path, stream=True)  # 流式处理视频

    for result in results:
        filename = f"{video_path}_frame_{result.frame}"  # 帧文件名（自定义）
        # 提取每个目标的信息
        # for box in result.boxes:
        #     class_id = int(box.cls[0])
        #     class_name = model.names[class_id]  # 目标类别名称
        #     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 边界框坐标
        #     confidence = float(box.conf[0])  # 置信度
        #     # 存储到数据库
        #     save_to_db(filename, class_name, x1, y1, x2, y2, confidence)
    print("数据库存储完成！")


# 测试
detect_and_save_to_db("test_video.mp4")