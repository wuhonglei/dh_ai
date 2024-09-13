import numpy as np
import matplotlib.pyplot as plt
# 实现rosenbrock函数的计算
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
# 定义自变量
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y) # 构造点阵
Z = rosenbrock(X, Y) # 计算函数值
#画出函数图像
board = plt.figure()
axis = board.add_subplot(1, 1, 1, projection='3d')
axis.plot_surface(X, Y, Z)
plt.show()

