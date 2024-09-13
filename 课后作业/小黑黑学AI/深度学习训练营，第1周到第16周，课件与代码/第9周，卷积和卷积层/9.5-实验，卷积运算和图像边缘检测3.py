
import numpy as np

#函数传入输入张量X和卷积核张量K
#在函数中，计算X和K的互相关运算结果返回
def correlation2d(X, K):
    Xh, Xw = X.shape #保存输入数据的高度和宽度
    Kh, Kw = K.shape #保存卷积核的高度和宽度
    Yh = Xh - Kh + 1 #计算输出数据的高度
    Yw = Xw - Kw + 1 #计算输出数据的宽度
    Y = np.zeros((Yh, Yw)) #保存输出结果
    for i in range(Yh):
        for j in range(Yw):
            sub = X[i:i + Kh, j:j + Kw]
            Y[i, j] = (sub * K).sum()
            #print("i = %d j = %d" % (i, j))
            #print(sub)
            #print(Y[i, j])
            #print("")
    return Y #返回结果Y

from matplotlib import pyplot as plt

# 真实图片包含了RGB三个通道，需要将它转为一个灰色通道
def rgb_to_gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

if __name__ == '__main__':
    #使用plt.imread读取图片数据
    img = plt.imread('./catdog.jpg')
    plt.imshow(img)  #将它展示在面板中
    plt.show()

    #真实图片包含了RGB三个通道，需要将它转为一个灰色通道
    img = rgb_to_gray(img)
    # 卷积核kernel，通过该数组可以将图像的边缘显示出来
    # 拉普拉斯算子，它可以将相同值的一片数据计算为0
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    # 计算img和kernel的互相关运算
    img = correlation2d(img, kernel)
    fig = plt.imshow(img)
    # 将它展示出来，可以发现正方形的边缘显示出来了
    plt.show()

