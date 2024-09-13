from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt
import numpy as np

# 函数传入num，代表每种类别的数据个数
def make_data(num):
    # 设定随机数生成器的种子
    # 使随机数序列，每次运行时都是确定的
    np.random.seed(0)
    # 红色数据使用make_blobs生成
    # 以(0, 0)为中心的正太分布数据
    red, _ = make_blobs(n_samples=num,
                        centers=[[0, 0]],
                        cluster_std=0.15)
    # 绿色数据用make_circles生成，分布在红色的周围
    green, _ = make_circles(n_samples=num,
                            noise=0.02,
                            factor=0.7)
    # 蓝色数据，数据分布在四个角落
    blue, _ = make_blobs(n_samples=num,
                           centers=[[-1.2, -1.2],
                                    [-1.2, 1.2],
                                    [1.2, -1.2],
                                    [1.2, 1.2]],
                           cluster_std=0.2)
    return red, green, blue #返回三种数据


if __name__ == '__main__':
    # 调用make_data，每种类别生成100个数据
    green, blue, red = make_data(100)
    # 创建-4到4的平面画板
    board = plt.figure()
    axis = board.add_subplot(1, 1, 1)
    axis.set(xlim=[-1.5, 1.5],
             ylim=[-1.5, 1.5],
             title='Neural Network',
             xlabel='x1',
             ylabel='x2')

    # 使用plt.scatter绘制出绿色、蓝色和红色三种数据
    plt.scatter(green[:, 0], green[:, 1], color='green')
    plt.scatter(blue[:, 0], blue[:, 1], color='blue')
    plt.scatter(red[:, 0], red[:, 1], color='red')
    plt.show()














