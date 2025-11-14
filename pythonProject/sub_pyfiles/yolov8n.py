""""
# 导出为TensorRT格式（.engine）
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="engine", imgsz=640)  # 仅需导出一次
# 加载加速模型
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from tkinter.filedialog import askopenfilenames
import cv2
from ultralytics import YOLO


def init_db():
    # 连接数据库（不存在则创建）
    conn = sqlite3.connect('detection_results.db')
    cursor = conn.cursor()
    # 创建表：存储检测时间、文件名、目标类别、坐标、置信度
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS detection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            frame_filename TEXT,
            class_name TEXT,
            x1 REAL, y1 REAL,
            x2 REAL, y2 REAL,
            confidence REAL
        )
                   ''')
    conn.commit()
    conn.close()


init_db()


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


# 加载预训练模型（yolov8n为轻量版，n/s/m/l/x代表模型大小递增）
model = YOLO("yolov8n.engine")

# 对图片进行检测，save=True表示保存结果
file_path = askopenfilenames(filetypes=[("Images", '*.webp *.jpg *.bmp *.dng *.png *.tiff *.jpeg *.pfm *.mpo *.tif *.heic'), ("Movies", '*.avi *.wmv *.gif *.m4v *.mpeg *.asf *.mkv *.mpg *.mov *.webm *.mp4 *.ts'), ("AllFiles", '*.*')])
if file_path:
    print("file_path:", file_path)
    for file in file_path:
        if Path(file).suffix in ('.mp4', '.webm', '.mov'):
            results = model(file, save=True, stream=True, verbose=False)
            frame_count = 1  # 初始化帧计数器
            cap = cv2.VideoCapture(file)
            for result in results:

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
                sys.stdout.write("\r")
                # 在处理每一帧的循环中
                print("\r", f"处理帧：{frame_count} of {total_frames}", end="", flush=True)
                frame_count += 1  # 递增计数器
                frame_filename = f"帧：{frame_count} of {total_frames}"
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]  # 目标类别名称
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # 边界框坐标
                    confidence = float(box.conf[0])  # 置信度
                    # 存储到数据库
                    save_to_db(frame_filename, class_name, x1, y1, x2, y2, confidence)
            print("\n")
        else:
            results = model(file, save=True)
            list(results)

# for result in results:
#     boxes = result.boxes  # 边界框信息
#     for box in boxes:
#         cls = box.cls  # 类别索引
#         conf = box.conf  # 置信度
#         xyxy = box.xyxy  # 边界框坐标（x1,y1,x2,y2）
#         print(f"类别：{model.names[int(cls)]}，置信度：{conf}，坐标：{xyxy}")
#
# img = cv2.imread(f"{results[0].save_dir}/{os.path.splitext(os.path.basename(file_path[0]))[0]}.jpg")
# cv2.imshow("Detection Result", img)
# cv2.waitKey(0)  # 等待按键（0 表示无限等待）
# cv2.destroyAllWindows()
