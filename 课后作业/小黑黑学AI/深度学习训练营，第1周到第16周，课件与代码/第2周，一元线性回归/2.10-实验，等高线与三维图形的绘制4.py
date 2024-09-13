import matplotlib.pyplot as plt
import numpy

if __name__ == '__main__':
    #生成等间距的列表x和列表y
    #生成0到3，间隔为1的列表x和y
    x = numpy.arange(0, 3.1, 1)
    y = numpy.arange(0, 3.1, 1)
    print('x = ', x)
    print('y = ', y)
    #使用meshgrid将x和y组合成网格矩阵
    #组合为4乘4网格矩阵
    x, y = numpy.meshgrid(x, y)
    print('x = ', x)
    print('y = ', y)
    #画出网格点
    plt.plot(x, y, marker='.',  markersize=10, linestyle='')
    plt.show()


