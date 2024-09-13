import numpy as np
import matplotlib.pyplot as plt

# 函数传入num，代表每种类别的数据个数
def make_data(num):
    # 使用np.random.randn生成满足正太分布的数据
    # 将生成的数据加上向量[0, -2]，代表绿色数据会以(0, -2)为中心分布
    green = np.random.randn(num, 2) + np.array([0, -2])
    # 生成蓝色数据，以(-2, 2)为中心分布，标准差为1的正太分布数据
    blue = np.random.randn(num, 2) + np.array([-2, 2])
    # 生成红色数据，以(2, 2)为中心分布，标准差为1的正太分布数据
    red = np.random.randn(num, 2) + np.array([2, 2])
    return green, blue, red

if __name__ == '__main__':
    # 调用make_data，每种类别生成30个数据
    green, blue, red = make_data(30)
    # 创建-4到4的平面画板
    board = plt.figure()
    axis = board.add_subplot(1, 1, 1)
    axis.set(xlim=[-4, 4],
             ylim=[-4, 4],
             title='Softmax Regression',
             xlabel='x1',
             ylabel='x2')

    # 使用plt.scatter绘制出绿色、蓝色和红色三种数据
    plt.scatter(green[:, 0], green[:, 1], color='green')
    plt.scatter(blue[:, 0], blue[:, 1], color='blue')
    plt.scatter(red[:, 0], red[:, 1], color='red')
    plt.show()



