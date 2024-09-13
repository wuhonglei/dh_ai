
import numpy as np

#函数传入输入张量X和卷积核张量K
#在函数中，计算X和K的互相关运算结果返回
def correlation2d(X, K):
    Xh, Xw = X.shape #保存输入数据的高度和宽度
    Kh, Kw = K.shape #保存卷积核的高度和宽度
    Yh = Xh - Kh + 1 #计算输出数据的高度
    Yw = Xw - Kw + 1 #计算输出数据的宽度
    Y = np.zeros((Yh, Yw)) #保存输出结果
    #通过两层循环，计算出数组Y的值
    for i in range(Yh): #第一层循环Y的高度
        for j in range(Yw): #第二层循环Y的宽度
            #取出对应位置的输入数据X，保存到sub中
            sub = X[i:i + Kh, j:j + Kw]
            #计算输出结果Y
            Y[i, j] = (sub * K).sum()
            #打印出当前的i、j、sub和Y，进行调试
            print("i = %d j = %d" % (i, j))
            print(sub)
            print(Y[i, j])
            print("")
    return Y #返回结果Y

if __name__ == '__main__':
    #测试函数的效果
    X = np.array([[1, 2, 3, 0],
                  [0, 1, 2, 3],
                  [3, 0, 1, 2],
                  [2, 3, 0, 1]])
    K = np.array([[2, 0, 1],
                  [0, 1, 2],
                  [1, 0, 2]])
    #观察到每一次互相关运算的结果
    #最后的输出数据是一个2乘2的矩阵
    result = correlation2d(X, K)
    print(f'result :\n {result}')

