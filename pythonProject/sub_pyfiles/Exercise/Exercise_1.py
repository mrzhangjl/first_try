from tkinter.filedialog import askopenfilename

import cv2

img = cv2.imread(askopenfilename(filetypes=[("JPEG", ".jpg"), ("PNG", ".png")]))
print("图片形状（高度、宽度、通道数）：", img.shape)  # 例如 (480, 640, 3) 表示高480px，宽640px，3通道（BGR）
print("图片数据类型：", img.dtype)  # 通常是 uint8（0-255的整数，最常见的图像格式）
print("图片总像素数：", img.size)  # 高度×宽度×通道数，即总元素数量

y, x = 100, 200
b, g, r = img[y, x]  # 分别获取B、G、R通道的像素值（0-255）
print(f"坐标({x}, {y})的像素值：B={b}, G={g}, R={r}")

conversion_types = [
    # BGR相关转换（OpenCV默认读取格式为BGR）
    (cv2.COLOR_BGR2GRAY, "COLOR_BGR2GRAY"),
    (cv2.COLOR_BGR2RGB, "COLOR_BGR2RGB"),
    (cv2.COLOR_BGR2HSV, "COLOR_BGR2HSV"),
    (cv2.COLOR_BGR2HLS, "COLOR_BGR2HLS"),
    (cv2.COLOR_BGR2YCrCb, "COLOR_BGR2YCrCb"),
    (cv2.COLOR_BGR2XYZ, "COLOR_BGR2XYZ"),
    (cv2.COLOR_BGR2LAB, "COLOR_BGR2LAB"),
    (cv2.COLOR_BGR2LUV, "COLOR_BGR2LUV"),

    # BGRA相关转换（带alpha通道）
    (cv2.COLOR_BGR2BGRA, "COLOR_BGR2BGRA"),
    (cv2.COLOR_BGRA2BGR, "COLOR_BGRA2BGR"),
    (cv2.COLOR_BGRA2GRAY, "COLOR_BGRA2GRAY"),

    # 反向转换（验证转换可逆性）
    (cv2.COLOR_RGB2BGR, "COLOR_RGB2BGR"),
    (cv2.COLOR_HSV2BGR, "COLOR_HSV2BGR"),
    (cv2.COLOR_LAB2BGR, "COLOR_LAB2BGR"),

    # 灰度图相关转换
    (cv2.COLOR_GRAY2BGR, "COLOR_GRAY2BGR"),
    (cv2.COLOR_GRAY2BGRA, "COLOR_GRAY2BGRA")
]


for code, name in conversion_types:
    try:
        # 执行颜色空间转换
        converted_img = cv2.cvtColor(img, code)

        # 对于单通道图像（如灰度图），为了统一显示效果可转为BGR（可选）
        # if len(converted_img.shape) == 2:
        #     converted_img = cv2.cvtColor(converted_img, cv2.COLOR_GRAY2BGR)

        # 显示转换结果（窗口名用功能描述）
        cv2.imshow(name, converted_img)
        print(f"已处理：{name}")
    except Exception as e:
        print(f"处理 {name} 失败：{str(e)}")

cv2.waitKey(0)
cv2.destroyAllWindows()