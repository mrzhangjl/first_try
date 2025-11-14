# OpenCV (cv2) 详细使用教程与示例代码

OpenCV（Open Source Computer Vision Library）是一个开源的计算机视觉和机器学习软件库。下面是一个详细的教程和示例代码。

## 目录
1. [安装与配置](#安装与配置)
2. [基础操作](#基础操作)
3. [图像处理](#图像处理)
4. [特征检测](#特征检测)
5. [视频处理](#视频处理)
6. [实战项目](#实战项目)

## 安装与配置

```bash
pip install opencv-python
# 如果需要额外模块
pip install opencv-contrib-python
```

## 基础操作

### 1. 读取和显示图像

```python
import cv2
import numpy as np

# 读取图像
img = cv2.imread('image.jpg')

# 显示图像
cv2.imshow('Image', img)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭所有窗口

# 获取图像信息
print(f"图像形状: {img.shape}")  # (高度, 宽度, 通道数)
print(f"图像尺寸: {img.size}")   # 总像素数
print(f"图像数据类型: {img.dtype}")  # 数据类型
```

### 2. 保存图像

```python
# 保存图像
cv2.imwrite('output.jpg', img)

# 转换颜色空间并保存
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.jpg', gray_img)
```

### 3. 绘制图形

```python
# 创建空白图像
canvas = np.zeros((500, 500, 3), dtype=np.uint8)

# 绘制矩形
cv2.rectangle(canvas, (50, 50), (200, 200), (0, 255, 0), 2)

# 绘制圆形
cv2.circle(canvas, (300, 300), 50, (255, 0, 0), -1)  # -1表示填充

# 绘制直线
cv2.line(canvas, (100, 400), (400, 400), (0, 0, 255), 3)

# 添加文字
cv2.putText(canvas, 'OpenCV Demo', (150, 100), 
           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow('Drawing', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 图像处理

### 1. 图像变换

```python
# 缩放
resized = cv2.resize(img, (300, 300))

# 旋转
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 45, 1.0)  # 旋转45度
rotated = cv2.warpAffine(img, matrix, (w, h))

# 透视变换
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, matrix, (300, 300))
```

### 2. 图像滤波

```python
# 高斯模糊
blurred = cv2.GaussianBlur(img, (15, 15), 0)

# 中值滤波
median = cv2.medianBlur(img, 5)

# 双边滤波（保持边缘）
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
```

### 3. 边缘检测

```python
# Canny边缘检测
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)  # 阈值1, 阈值2

# Sobel算子
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobelx, sobely)
```

### 4. 阈值处理

```python
# 简单阈值
ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 自适应阈值
thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 11, 2)

# Otsu's二值化
ret, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

## 特征检测

### 1. 角点检测

```python
# Harris角点检测
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 标记角点为红色

# Shi-Tomasi角点检测
corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)
```

### 2. SIFT特征检测

```python
# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 检测关键点和描述符
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 绘制关键点
img_sift = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
```

### 3. 特征匹配

```python
# 读取两张图像
img1 = cv2.imread('image1.jpg', 0)
img2 = cv2.imread('image2.jpg', 0)

# 初始化特征检测器
orb = cv2.ORB_create()

# 找到关键点和描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 创建BFMatcher对象
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 匹配描述符
matches = bf.match(des1, des2)

# 按距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 绘制前10个匹配
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
```

## 视频处理

### 1. 读取和显示视频

```python
# 读取视频文件
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 处理帧（这里可以添加各种图像处理操作）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Video', gray)
    
    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 2. 摄像头实时处理

```python
# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # 边缘检测
    edges = cv2.Canny(frame, 100, 200)
    
    cv2.imshow('Camera', frame)
    cv2.imshow('Edges', edges)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 3. 保存视频

```python
# 定义编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        # 写入帧
        out.write(frame)
        
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

## 实战项目

### 1. 人脸检测

```python
# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # 绘制矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image

# 实时人脸检测
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        result = detect_faces(frame)
        cv2.imshow('Face Detection', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

### 2. 图像拼接（全景图）

```python
def stitch_images(images):
    stitcher = cv2.Stitcher.create()
    status, panorama = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        return panorama
    else:
        print("图像拼接失败")
        return None

# 读取多张图像
images = []
for i in range(1, 4):
    img = cv2.imread(f'image{i}.jpg')
    images.append(img)

# 拼接图像
result = stitch_images(images)
if result is not None:
    cv2.imshow('Panorama', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### 3. 文档扫描仪

```python
def document_scanner(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 边缘检测
    edged = cv2.Canny(blurred, 75, 200)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # 寻找文档轮廓
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            screen_cnt = approx
            break
    
    # 透视变换
    warped = four_point_transform(image, screen_cnt.reshape(4, 2))
    
    return warped

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect
```

## 总结

这个教程涵盖了OpenCV的主要功能，包括：

- **基础操作**：图像读取、显示、保存、绘制
- **图像处理**：变换、滤波、边缘检测、阈值处理
- **特征检测**：角点检测、SIFT、特征匹配
- **视频处理**：视频读写、摄像头操作
- **实战项目**：人脸检测、图像拼接、文档扫描

OpenCV是一个功能强大的计算机视觉库，掌握这些基础操作后，你可以进一步探索更高级的功能，如机器学习、深度学习集成、3D重建等。建议多动手实践，结合具体项目来加深理解。