import cv2
import numpy as np

img_path = './data/train/0_1008.png'
gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 二值化处理，将黑色区域设为 1，其他为 0
_, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY_INV)

# 创建一个核，用于形态学操作
kernel = np.ones((3, 3), np.uint8)

# 使用闭运算去除黑色噪点
cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# 反转颜色，恢复到原来的背景色
cleaned_image = cv2.bitwise_not(cleaned_image)

# 将处理后的二值图像应用到原始图像上
result = cv2.bitwise_and(gray_image, gray_image, mask=cleaned_image)


# 显示原始图像和结果
cv2.imshow('Original Image', gray_image)
cv2.imshow('Cleaned Image', result)

key = cv2.waitKey(0)
if key == ord('q'):  # ESC
    cv2.destroyWindow('img')
