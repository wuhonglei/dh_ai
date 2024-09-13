# 导入pytorch、numpy和绘图库
import torch
import numpy
import matplotlib.pyplot as plt

# 定义函数linear_model，计算一元线性回归
# 其中x是输入特征，w和b是待迭代的参数
def linear_model(x, w, b):
    return x * w + b

# 定义函数mse_loss，计算均方误差
# 其中h是直线的预测值，y是真实值
def mse_loss(h, y):
    return torch.mean((h - y) ** 2)

if __name__ == '__main__':
    # 定义张量x和张量y，分别保存房屋的面积特征和真实的价格
    x = torch.tensor([50.0, 60.0, 70.0, 80.0,
                      90.0, 100.0, 110.0, 120.0])
    y = torch.tensor([280.0, 305.0, 350.0, 425.0,
                      480.0, 500.0, 560.0, 630.0])

    # 使用torch.randn初始化直线的参数w和b
    # 将requires_grad设置为True，计算梯度
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)

    learning_rate = 0.0001 # 设置学习率
    n = 1000 # 迭代次数

    for i in range(n): #进入梯度下降的循环迭代
        h = linear_model(x, w, b) #计算当前直线的预测值
        # 调用函数mse_loss，计算预测值h和真实值y之间的均方误差
        loss = mse_loss(h, y)

        loss.backward() #计算代价loss关于参数w和b的偏导数

        # 进行梯度下降，沿着梯度的反方向，更新w和b的值
        # w和b的偏导数保存在grad.data中，更新w.data和b.data
        w.data -= learning_rate * w.grad.data
        b.data -= learning_rate * b.grad.data

        # 清空张量w和b中的梯度信息，为下一次迭代做准备
        w.grad.zero_()
        b.grad.zero_()

        if i % 100 == 0: #每100次迭代，打印一次loss损失值
            print(f'Epoch {i}, Loss: {loss.item()}')

    print('w = %.3lf, b = %.3lf'%(w.item(), b.item()))

    # 将张量w和b转为普通的变量theta1和theta0
    theta0 = b.item()
    theta1 = w.item()

    # 使用可视化的方式，将样本数据与迭代出的直线展示出来
    # 绘制图像的代码和前面课程中的是一样的
    board = plt.figure()  # 创建一个figure画板对象board
    # 从画板对象中分割出一个1行1列的区域
    # 并取出该区域保存至变量axis
    axis = board.add_subplot(1, 1, 1)
    # 设置一个x轴为0到150、y轴为0到800的坐标系
    # 该坐标系的标题是Linear Regression
    # x轴的名称是area，y轴的名称是price
    axis.set(xlim=[0, 150],
             ylim=[0, 800],
             title='Linear Regression',
             xlabel='area',
             ylabel='price')
    # 传入样本坐标列表、点的标记颜色和形状加号
    # 在画板上画出了8个使用加号表示的独立样本
    plt.scatter(x, y, color='red', marker='+')
    # 在0到150之间，构造出500个相同间距的浮点数，保存至x
    x = numpy.linspace(0, 150, 500)
    h = theta1 * x + theta0  # 直线的函数值
    plt.plot(x, h)  # 画出f1的图像
    plt.show()  # 调用show展示，就会得到一个空的画板

