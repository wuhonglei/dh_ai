import matplotlib.pyplot as plt
import numpy

if __name__ == '__main__':
    board = plt.figure()  # 创建画板对象
    # 生成坐标轴，需要传入参数projection，等于3d
    axis = board.add_subplot(1, 1, 1, projection='3d')
    #生成-2到2，间隔为0.5的列表x和y，对应横轴和纵轴上的坐标
    X = numpy.arange(-2, 2.1, 0.5)
    Y = numpy.arange(-2, 2.1, 0.5)

    # 如果我们将x和y的坐标间隔修改为0.1
    # 重新生成图像，会发现图像上的网格变得更密集了
    # X = numpy.arange(-2, 2.1, 0.1)
    # Y = numpy.arange(-2, 2.1, 0.1)

    #通过meshgrid生成坐标矩阵
    X, Y = numpy.meshgrid(X, Y)
    #根据X和Y，计算Z的值
    Z = X ** 2 + Y ** 2
    axis.plot_surface(X, Y, Z) #绘制出Z的图像
    plt.show()

