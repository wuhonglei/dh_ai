from sklearn.datasets import make_blobs, make_circles
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

# 函数传入num，代表每种类别的数据个数
def make_data(num):
    # 设定随机数生成器的种子，使随机数序列，每次运行时都是确定的
    np.random.seed(0)
    # 红色数据使用make_blobs生成，是以(0, 0)为中心的正太分布数据
    red, _ = make_blobs(n_samples=num,
                        centers=[[0, 0]],
                        cluster_std=0.15)
    # 绿色数据使用make_circles生成，分布在红色数据的周围
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

# 定义神经网络类Network，它继承nn.Module类
class Network(nn.Module):
    # 函数传入参数n_in, n_hidden, n_out
    # 代表输入层、隐藏层和输出层中的神经元数量
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__() # 调用了父类的初始化函数
        # layer1是输入层与隐藏层之间的线性层
        self.layer1 = nn.Linear(n_in, n_hidden)
        # layer2是隐藏层与输出层之间的线性层
        self.layer2 = nn.Linear(n_hidden, n_out)

    # 实现神经网络的前向传播，函数传入输入数据x
    def forward(self, x):
        x = self.layer1(x)  # 计算layer1的结果
        x = torch.relu(x)  # 进行relu激活
        # 计算layer2的结果，并返回
        return self.layer2(x)



# 生成用于绘制决策边界的等高线数据
# min-x1到max-x1是画板的横轴范围，min-x2到max-x2是画板的纵轴范围
# model是训练好的模型
# 函数中，我们会根据已训练的model，计算对应类别结果，
# 不同类别结果会对应不同的高度
# 基于数据点的坐标与高度数据，绘制等高线
def draw_decision_boundary(minx1, maxx1, minx2, maxx2, model):
    # 调用mesh-grid生成网格数据点
    # 每个点的距离是0.02，这样生成的点可以覆盖平面的全部范围
    xx1, xx2 = np.meshgrid(np.arange(minx1, maxx1, 0.02),
                           np.arange(minx2, maxx2, 0.02))
    # 设置x1s、x2s和z分别表示数据点的横坐标、纵坐标和预测结果
    x1s = xx1.ravel()
    x2s = xx2.ravel()
    z = list()
    for x1, x2 in zip(x1s, x2s):  # 遍历全部样本
        # 将样本转为张量
        test_point = torch.FloatTensor([[x1, x2]])
        output = model(test_point)  # 使用model预测结果
        # 选择概率最大的类别
        _, predicted = torch.max(output, 1)
        z.append(predicted.item())  # 添加到高度z中
    # 将z重新设置为和xx1相同的形式
    z = np.array(z).reshape(xx1.shape)
    return xx1, xx2, z  # 返回xx1、xx2和z


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

    n_features = 2  # 特征数
    n_hidden = 5 # 隐藏层神经元个数
    n_classes = 3  # 类别数
    n_epochs = 10000  # 迭代次数
    learning_rate = 0.001  # 学习速率

    # 将绿色、蓝色、红色三种样本，从numpy数组转换为张量形式
    green = torch.FloatTensor(green)
    blue = torch.FloatTensor(blue)
    red = torch.FloatTensor(red)
    # 一起组成训练数据data
    data = torch.cat((green, blue, red), dim=0)
    # 设置label保存三种样本的标签
    label = torch.LongTensor([0] * len(green) +
                             [1] * len(blue) +
                             [2] * len(red))

    # 创建神经网络模型实例
    model = Network(n_features, n_hidden, n_classes)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # Adam优化器optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)


    # 进入softmax回归模型的循环迭代
    for epoch in range(n_epochs):

        # 使用当前的模型，预测训练数据data，结果保存在output中
        output = model(data)

        # 计算预测值output与真实值label之间的损失loss
        loss = criterion(output, label)
        loss.backward()  # 通过自动微分计算损失函数关于模型参数的梯度
        optimizer.step()  # 更新模型参数，使得损失函数减小
        optimizer.zero_grad()  # 将梯度清零，以便于下一次迭代

        # 模型的每一轮迭代，有前向传播和反向传播共同组成
        if epoch % 1000 == 0:
            # 每1000次迭代，打印一次当前的损失
            print('%d iterations : loss = %.3lf'
                  % (epoch, loss.item()))


    # 使用函数draw_decision_boundary，生成数据
    xx1, xx2, z = draw_decision_boundary(-4, 4, -4, 4, model)
    # 调用plt.contour绘制多分类的决策边界
    plt.contour(xx1, xx2, z, colors=['orange'])
    plt.show()




















