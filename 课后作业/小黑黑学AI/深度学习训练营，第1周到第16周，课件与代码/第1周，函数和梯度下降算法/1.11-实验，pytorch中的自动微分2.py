# 导入torch库和plt绘图库
import torch
import matplotlib.pyplot as plt

# 定义函数f，计算二次函数f(x) = x ^ 2 + 3 * x + 2的值
def f(x):
    return x * x + 3 * x + 2

if __name__ == '__main__':
    # 使用torch.linspace生成x值，并设置requires_grad=True
    x = torch.linspace(-6.5, 3.5, 1000, requires_grad=True)
    y_f = f(x)  # 计算函数f的值，保存到y_f中

    # 使用backward函数，计算y_f.sum()关于x的梯度，梯度值会保存在x.grad中
    y_f.sum().backward()
    # 这里要注意，因为backward()函数只能对标量进行操作，
    # 所以需要将y_f中的所有元素求和，将其转换为一个标量，
    # 然后在这个标量上调用backward()计算梯度。

    # 将梯度值x.grad、函数值y_f、自变量x，从pytorch张量转换为numpy数组
    y_df = x.grad.detach().numpy()
    y_f = y_f.detach().numpy()
    x = x.detach().numpy()
    # 因为numpy()方法不能直接在需要梯度的张量上调用，
    # 所以使用detach方法创建了一个与原张量无关的副本，
    # 从而将张量转换为numpy数组。

    # 使用plot绘制图
    plt.plot(x, y_f, label='f(x) = x * x + 3 * x + 2')
    plt.plot(x, y_df, label="f'(x) = 2 * x + 3")
    plt.legend()  # 对图像进行标记
    plt.grid(True)  # 使用grid函数标记出网格线
    plt.show()  # 展示图像

