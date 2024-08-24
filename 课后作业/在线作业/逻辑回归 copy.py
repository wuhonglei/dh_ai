import numpy
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import torch
from torch import nn


class Network(nn.Module):
    # 模型的初始化init函数
    def __init__(self):
        super(Network, self).__init__()
        # 神经网络有两个线性层
        self.hidden = nn.Linear(in_features=2,
                                out_features=3)

        self.out = nn.Linear(in_features=3,
                             out_features=2)

    # 前向传播forward函数
    def forward(self, x):
        x = self.hidden(x)  # 增加了隐藏层的计算
        x = torch.sigmoid(x)
        x = self.out(x)
        return x

# 生成用于绘制决策边界的等高线数据
# min-x1到max-x1是画板的横轴范围，min-x2到max-x2是画板的纵轴范围
# model是训练好的模型
# 函数中，我们会根据已训练的model，计算对应类别结果，
# 不同类别结果会对应不同的高度
# 基于数据点的坐标与高度数据，绘制等高线


def draw_decision_boundary(minx1, maxx1, minx2, maxx2, model):
    # 调用mesh-grid生成网格数据点
    # 每个点的距离是0.02，这样生成的点可以覆盖平面的全部范围
    xx1, xx2 = numpy.meshgrid(numpy.arange(minx1, maxx1, 0.02),
                              numpy.arange(minx2, maxx2, 0.02))
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
    z = numpy.array(z).reshape(xx1.shape)
    return xx1, xx2, z  # 返回xx1、xx2和z


if __name__ == '__main__':
    # 使用make_blobs函数，在平面上生成50个随机样本，包含两个类别
    x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=1)
    # x1和x2分别保存样本的两个特征
    x1 = x[:, 0]
    x2 = x[:, 1]
    # 使用plt.scatter绘制正样本和负样本
    plt.scatter(x1[y == 1], x2[y == 1], color='blue', marker='o')
    plt.scatter(x1[y == 0], x2[y == 0], color='red', marker='x')

    # 将x和y转换为张量形式
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = Network()  # 定义模型
    optimizer = torch.optim.Adam(model.parameters())  # 定义Adam优化器
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10000):  # 进入循环迭代，将所有的数据，一起迭代
        optimizer.zero_grad()  # 将梯度清零

        # 前向传播
        predict = model(x_tensor)  # 计算模型的预测值
        # 前向传播结束

        # 反向传播
        # 计算predict和y_tensor之间的损失loss
        loss = criterion(predict, y_tensor)
        loss.backward()  # 计算loss关于参数的梯度
        optimizer.step()  # 更新模型参数
        # 反向传播结束

        if epoch % 1000 == 0:  # 每1000次迭代，打印一次当前的损失
            # loss.item是损失的标量值
            print(f'After {epoch} iterations, the loss is {loss.item():.3f}')

    # draw_decision_boundary是一个自定义的绘制边界的函数
    xx1, xx2, z = draw_decision_boundary(-2, 6, -2, 6, model)
    # 绘制橙色的决策边界，原理是将决策边界看做是等高线
    plt.contour(xx1, xx2, z, colors=['orange'])
    plt.show()
