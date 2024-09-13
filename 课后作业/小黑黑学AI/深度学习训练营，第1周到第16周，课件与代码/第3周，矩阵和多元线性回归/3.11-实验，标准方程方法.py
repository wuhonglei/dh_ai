import numpy

#函数传入特征向量矩阵和标签列向量y
def normal_equation_array(X, y):
    XT = numpy.transpose(X) #调用transpose计算矩阵X的转置
    XTX = numpy.dot(XT, X) #调用dot计算X转置乘X
    det = numpy.linalg.det(XTX) #计算结果矩阵的行列式的值
    #在行列式值不为0的情况下，才可以进行求逆矩阵的运算
    if det == 0.0:
        return "error"
    XTX_inv = numpy.linalg.inv(XTX) #计算XTX的逆矩阵
    XTy = numpy.dot(XT, y) #计算X转置和y相乘
    theta = numpy.dot(XTX_inv, XTy) #计算theta矩阵
    return theta

#直接使用numpy中的矩阵类，使用起来更简单方便
def normal_equation_matrix(X, y):
    #将数组X和y转换为矩阵
    X = numpy.mat(X)
    y = numpy.mat(y)
    #矩阵X，可以直接调用X点T来得到X的转置
    XTX = X.T * X #矩阵的相乘可以直接使用乘法符号*号
    det = numpy.linalg.det(XTX)
    if det == 0.0:
        return "error"
    #X点I得到X的逆矩阵，根据公式，计算出theta矩阵
    theta = XTX.I * X.T * y
    return theta

if __name__ == '__main__':
    #设置特征向量矩阵X
    X = numpy.array([[1, 96.79, 2, 1, 2],
                    [1, 110.39, 3, 1, 0],
                    [1, 70.25, 1, 0, 2],
                    [1, 99.96, 2, 1, 1],
                    [1, 118.15, 3, 1, 0],
                    [1, 115.08, 3, 1, 2]])
    #设置标签列向量y
    y = numpy.array([[287],
                     [343],
                     [199],
                     [298],
                     [340],
                     [350]])
    #测试两个函数，求出theta的值
    theta = normal_equation_array(X, y)
    print(theta)
    theta = normal_equation_matrix(X, y)
    print(theta)

    #计算代价J
    costJ = (y - X * theta).T * (y - X * theta) / (2 * len(y))
    print("Cost J is %lf" % (costJ))
    # 预测两个样本结果
    test1 = [1, 112, 3, 1, 0]
    test2 = [1, 110, 3, 1, 1]
    # 打印两个测试样本的结果
    print("test1 = %.3f" % (test1 * theta))
    print("test2 = %.3f" % (test2 * theta))


