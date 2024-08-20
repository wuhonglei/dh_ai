import matplotlib.pyplot as plt
import numpy as np


def draw_clear_board(x_min, x_max, y_min, y_max):
    fig, ax = plt.subplots()

    # 移动 x 轴和 y 轴到图形中心
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # 隐藏顶部和右侧的脊线
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # 在 x 轴和 y 轴上添加箭头
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    # 添加网格
    # ax.grid(True, which='both')

    # 设置 x 和 y 轴的范围
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.show()
