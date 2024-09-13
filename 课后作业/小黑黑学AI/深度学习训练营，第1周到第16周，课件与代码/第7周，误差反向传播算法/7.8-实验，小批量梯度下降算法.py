import numpy as np
import matplotlib.pyplot as plt

# 使用random.seed，设置一个固定的随机种子，
# 可以确保每次运行，都得到相同的数据，方便调试
np.random.seed(42)
# 随机生成100个横坐标x，范围在0到2之间
x = 2 * np.random.rand(100, 1)
# 生成带有噪音的纵坐标y，数据基本分布在y=4+3x的附近
y = 4 + 3 * x + np.random.randn(100, 1) * 0.5

plt.scatter(x, y, marker='x', color='red')

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# 将训练数据x和y转为张量
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()

# 使用TensorDataset，将x和y组成训练集
dataset = TensorDataset(x, y)
# 使用DataLoader，构造随机的小批量数据
dataloader = DataLoader(dataset,
                        # 每一个小批量的数据规模是16
                        batch_size = 16,
                        # 随机打乱数据的顺序
                        shuffle = True)

print("dataloader len = %d" % (len(dataloader)))
for index, (data, label) in enumerate(dataloader):
    print("index = %d num = %d"%(index, len(data)))


# 待迭代的参数为w和b
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
# 进入模型的循环迭代
for epoch in range(1, 51): # 代表了整个训练数据集的迭代轮数
    # 在一个迭代轮次中，以小批量的方式，使用dataloader对数据进行遍历
    # batch_idx表示当前遍历的批次
    # data和label表示这个批次的训练数据和标记
    for batch_idx, (data, label) in enumerate(dataloader):
        h = x * w + b # 计算当前直线的预测值，保存到h
        # 计算预测值h和真实值y之间的均方误差，保存到loss中
        loss =  torch.mean((h - y) ** 2)
        loss.backward() # 计算代价loss关于参数w和b的偏导数
        # 进行梯度下降，沿着梯度的反方向，更新w和b的值
        w.data -= 0.01 * w.grad.data
        b.data -= 0.01 * b.grad.data
        # 清空张量w和b中的梯度信息，为下一次迭代做准备
        w.grad.zero_()
        b.grad.zero_()
        # 每次迭代，都打印当前迭代的轮数epoch
        # 数据的批次batch_idx和loss损失值
        print("epoch(%d) batch(%d) loss = %.3lf"
                %(epoch, batch_idx, loss.item()))


# 打印w和b的值，并绘制直线
print('w = %.3lf, b = %.3lf'%(w.item(), b.item()))
w = w.item()
b = b.item()
x = np.linspace(0, 2, 100)
h = w * x + b
plt.plot(x, h)
plt.show()

