import torch
from torch import nn

# 神经网络类Network，继承nn.Module类
class Network(nn.Module):
    # 数传入参数n_in, n_hidden, n_out
    # 代表输入层、隐藏层和输出层中的特征数量
    def __init__(self, n_in, n_hidden, n_out):
        # 调用了父类 torch.nn.Module 的初始化函数
        super().__init__()
        # layer1是输入层与隐藏层之间的线性层
        self.layer1 = nn.Linear(n_in, n_hidden)
        # layer2是隐藏层与输出层之间的线性层
        self.layer2 = nn.Linear(n_hidden, n_out)

    # 实现神经网络的前向传播，函数传入输入数据x
    def forward(self, x):
        x = self.layer1(x) #计算layer1的结果
        x = torch.sigmoid(x) #进行sigmoid激活
        return self.layer2(x) #计算layer2的结果，并返回

from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 使用np.arrange生成一个从0到1，步长为0.01
    # 含有100个数据点的数组，作为正弦函数的输入数据
    x = np.arange(0.0, 1.0, 0.01)
    # 将0到1的x，乘以2π，从单位间隔转换为弧度值
    # 将x映射到正弦函数的一个完整周期上，并计算正弦值
    y = np.sin(2 * np.pi * x)
    # 将x和y通过reshape函数转为100乘1的数组
    # 也就是100个(x, y)坐标，代表100个训练数据
    x = x.reshape(100, 1)
    y = y.reshape(100, 1)
    # 将(x, y)组成的数据点，画在画板上
    plt.scatter(x, y)

    x = torch.Tensor(x) # 将数据x和y转化为tensor张量
    y = torch.Tensor(y)
    model = Network(1, 10, 1) #定义一个3层的神经网络model
    criterion = nn.MSELoss() #创建均方误差损失函数
    # Adam优化器optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10000): #进入神经网络模型的循环迭代
        y_pred = model(x) #使用当前的模型，预测训练数据x

        #计算预测值y_pred与真实值y_tensor之间的损失
        loss = criterion(y_pred, y)
        loss.backward() #通过自动微分计算损失函数关于模型参数的梯度
        optimizer.step() #更新模型参数，使得损失函数减小
        optimizer.zero_grad() #将梯度清零，以便于下一次迭代

        #模型的每一轮迭代，都由前向传播和反向传播共同组成
        if epoch % 1000 == 0:
            #每1000次迭代，打印一次当前的损失
            print(f'After {epoch} iterations, the loss is {loss.item()}')

    h = model(x) # 使用模型预测输入x，得到预测结果h
    x = x.data.numpy()
    h = h.data.numpy()
    plt.scatter(x, h)  # 将预测点(x, h)打印在屏幕
    plt.show()




