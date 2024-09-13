# 定义函数f，计算二次函数f(x) = x ^ 2 + 3 * x + 2的值
def f(x):
    return x * x + 3 * x + 2

# 手动求出f关于x的导数函数，结果是2 * x + 3
def df(x):
    return 2 * x + 3

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 使用linspace，生成自变量x的值
    x = np.linspace(-6.5, 3.5, 1000)
    y_f = f(x) # 计算函数f的值
    y_df = df(x) # 计算函数df的值
    # 使用plot绘制图
    plt.plot(x, y_f, label='f(x) = x * x + 3 * x + 2')
    plt.plot(x, y_df, label="f'(x) = 2 * x + 3")
    plt.legend() # 对图像进行标记
    plt.grid(True) # 使用grid函数标记出网格线
    plt.show() # 展示图像

