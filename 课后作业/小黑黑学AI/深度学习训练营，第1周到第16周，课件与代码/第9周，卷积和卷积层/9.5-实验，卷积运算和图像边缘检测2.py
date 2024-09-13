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

def print_img(img):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            print("%5d"%(img[i][j]), end=" ")
        print("")

if __name__ == '__main__':
    #设置一个数组img，其中保存的数据代表了图像中的像素。
    img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                    [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                    [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                    [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                    [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                    [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    plt.imshow(img)  # 将它展示在面板中
    plt.show()

    #卷积核kernel，通过该数组可以将图像的边缘显示出来
    #拉普拉斯算子，它可以将相同值的一片数据计算为0
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    print_img(img)
    #计算img和kernel的互相关运算
    img = correlation2d(img, kernel)
    print_img(img)
    fig = plt.imshow(img)
    # 将它展示出来，可以发现正方形的边缘显示出来了
    plt.show()



