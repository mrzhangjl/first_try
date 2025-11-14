# python
import math
import cv2
import numpy as np
import sys
import os

def read_show_save(path):
    img = cv2.imread(path)
    if img is None:
        print("无法读取图像：", path)
        return
    print("shape:", img.shape, "dtype:", img.dtype)
    cv2.imshow("Original", img)
    out_path = os.path.splitext(path)[0] + "_out.jpg"
    cv2.imwrite(out_path, img)
    print("已保存：", out_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_conversion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("BGR", img)
    cv2.imshow("GRAY", gray)
    cv2.imshow("RGB", rgb)
    cv2.imshow("HSV (V channel)", hsv[:,:,2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_demo(img):
    canvas = img.copy()
    h, w = canvas.shape[:2]
    # 直线
    cv2.line(canvas, (0,0), (w-1, h-1), (0,255,0), 2)
    # 矩形
    cv2.rectangle(canvas, (10,10), (w//3, h//4), (255,0,0), 3)
    # 圆
    cv2.circle(canvas, (w//2, h//2), min(w,h)//6, (0,0,255), -1)
    # 文本
    cv2.putText(canvas, "OpenCV Demo", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Draw", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_rotate(img):
    # 缩放
    small = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    large = cv2.resize(img, (800, 600), interpolation=cv2.INTER_LINEAR)
    # 旋转（绕图像中心）
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), 30, math.cos(math.acos(w/math.sqrt(h*h+w*w))) / math.cos(math.acos(w/math.sqrt(h*h+w*w))-math.radians(30)))# 旋转30度
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imshow("Small", small)
    cv2.imshow("Large", large)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def threshold_blur_canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 简单阈值
    _, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # 自适应阈值
    ath = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # 模糊
    blur = cv2.GaussianBlur(img, (7,7), 1.5)
    # Canny 边缘
    edges = cv2.Canny(gray, 50, 150)
    cv2.imshow("Threshold", th)
    cv2.imshow("Adaptive Threshold", ath)
    cv2.imshow("Blur", blur)
    cv2.imshow("Canny Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contours_demo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    cv2.drawContours(out, contours, -1, (0,255,0), 2)
    print("轮廓数量：", len(contours))
    cv2.imshow("Contours", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def webcam_demo():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    print("按 q 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 鼠标回调示例：点击显示坐标与 BGR
def mouse_callback_demo(img):
    display = img.copy()
    window = "Mouse Demo"
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            b,g,r = img[y,x]
            print(f"坐标({x},{y}) B={b} G={g} R={r}")
            cv2.circle(display, (x,y), 3, (0,255,255), -1)
            cv2.imshow(window, display)
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    cv2.imshow(window, display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 可交互滑动条用于调整 Canny 参数
def canny_trackbar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    def nothing(x):
        pass
    cv2.namedWindow("Canny Trackbar")
    cv2.createTrackbar("Min", "Canny Trackbar", 50, 500, nothing)
    cv2.createTrackbar("Max", "Canny Trackbar", 150, 500, nothing)
    while True:
        mn = cv2.getTrackbarPos("Min", "Canny Trackbar")
        mx = cv2.getTrackbarPos("Max", "Canny Trackbar")
        if mn >= mx:
            mx = mn + 1
        edges = cv2.Canny(gray, mn, mx)
        cv2.imshow("Canny Trackbar", edges)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 2:
        print("用法：python cv2_tutorial.py 图像路径")
        print("也可运行 python cv2_tutorial.py webcam 进入摄像头演示")
        return
    arg = sys.argv[1]
    if arg.lower() == "webcam":
        webcam_demo()
        return
    if not os.path.exists(arg):
        print("文件不存在：", arg)
        return
    img = cv2.imread(arg)
    read_show_save(arg)
    color_conversion(img)
    draw_demo(img)
    resize_rotate(img)
    threshold_blur_canny(img)
    contours_demo(img)
    mouse_callback_demo(img)
    canny_trackbar(img)
    print("演示结束")

if __name__ == "__main__":
    main()