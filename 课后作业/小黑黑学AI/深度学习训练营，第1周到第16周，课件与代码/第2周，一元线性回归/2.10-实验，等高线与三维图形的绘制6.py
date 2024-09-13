import matplotlib.pyplot as plt
import numpy

if __name__ == '__main__':
    board = plt.figure()  # 创建画板对象
    # 生成坐标轴，需要传入参数projection，等于3d
    axis = board.add_subplot(1, 1, 1, projection='3d')
    # 生成-2到2，间隔为0.2的列表x和y，对应横轴和纵轴上的坐标
    X = numpy.arange(-2, 2.1, 0.2)
    Y = numpy.arange(-2, 2.1, 0.2)
    # 通过meshgrid生成坐标矩阵
    X, Y = numpy.meshgrid(X, Y)
    # 根据X和Y，计算Z的值
    Z = X ** 2 + Y ** 2
    #生成z等于x平方加y平方的三维等高线图像
    plt.contour(X, Y, Z)
    plt.show()

