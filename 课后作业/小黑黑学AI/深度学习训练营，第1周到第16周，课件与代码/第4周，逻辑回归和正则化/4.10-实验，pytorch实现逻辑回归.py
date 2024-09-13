
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy
import torch

# 基于nn.Module模块，实现逻辑回归模型
# 定义了类LogisticRegression，继承nn.Module类
class LogisticRegression(torch.nn.Module):
    # 类的初始化函数init，参数n_features代表输入特征的数量
    def __init__(self, n_features):
        # 调用了父类 torch.nn.Module 的初始化函数
        super(LogisticRegression, self).__init__()
        # 该线性层输入n_features个特征，输出1个结果
        self.linear = torch.nn.Linear(n_features, 1)

    #在forward函数中，实现逻辑回归的计算。函数传入输入数据x
    def forward(self, x):
        # 通过线性层linear计算线性变换的结果
        # 将结果通过sigmoid函数映射到(0,1)区间
        h = torch.sigmoid(self.linear(x))
        return h #返回结果h

if __name__ == '__main__':
    # 使用make_blobs，生成50个随机样本，包含两个类别
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.5)
    # 设置pos_mask和neg_mask保存正样本和负样本的索引
    pos_mask = y == 1
    neg_mask = y == 0
    plt.figure() #创建画板后，使用plt.scatter绘制正样本和负样本
    #其中正样本蓝色圆圈表示，负样本使用红色叉子表示
    plt.scatter(X[pos_mask, 0], X[pos_mask, 1], color='blue', marker='o')
    plt.scatter(X[neg_mask, 0], X[neg_mask, 1], color='red', marker='x')
    #接着再将坐标系绘制出来
    plt.xlim([-1, 6])
    plt.ylim([-1, 6])
    plt.title('Logistic Regression')
    plt.xlabel('x1')
    plt.ylabel('x2')

    # 将数据转化为tensor张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    # view(-1, 1)会将y从1乘50的行向量转为50乘1的列向量
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    m = len(X) #样本数量
    n = 2 #类别数量
    alpha = 0.001 #迭代速率
    iterate = 20000 #迭代次数
    model = LogisticRegression(n) #创建逻辑回归模型实例
    criterion = torch.nn.BCELoss() #二分类交叉熵损失函数
    #SGD优化器optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha)

    for epoch in range(iterate): #进入逻辑回归模型的循环迭代
        y_pred = model(X_tensor) # 使用当前的模型，预测训练数据
        # 计算预测值y_pred与真实值y_tensor之间的损失loss
        loss = criterion(y_pred, y_tensor)
        loss.backward() #通过自动微分计算损失函数关于模型参数的梯度
        optimizer.step() #更新模型参数，使得损失函数减小
        optimizer.zero_grad() #将梯度清零，以便于下一次迭代

        #模型的每一轮迭代，有前向传播和反向传播共同组成
        if epoch % 1000 == 0:
            #每1000次迭代，打印一次当前的损失
            print(f'After {epoch} iterations, the loss is {loss.item()}')

    # 取训练得到的逻辑回归模型参数
    # 并基于这些参数，通过可视化的方法绘制决策边界
    theta = list() #创建了一个空列表
    for p in model.parameters():
        #遍历模型的所有参数，添加到theta中
        theta.extend(p.detach().numpy().flatten())
    for i in range(0, len(theta)): #打印theta列表
        print("theta[%d] = %lf" % (i, theta[i]))

    #通过plt，绘制分类的决策边界
    w1 = theta[0]
    w2 = theta[1]
    b = theta[2]
    x = numpy.linspace(-1, 6, 100)
    d = - (w1 * x + b) * 1.0 / w2
    plt.plot(x, d)
    plt.show()

