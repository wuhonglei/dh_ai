import torch
from torch import nn

# 神经网络类Network，继承nn.Module类
class Network(nn.Module):
    # 实现神经网络的3个线性层，并手动的对线性层中的w和b进行赋值
    def __init__(self):
        super().__init__()
        # 使用nn.Linear，定义第1个线性层fc1
        self.fc1 = nn.Linear(2, 3) # fc1包括了2个输入特征和3个输出特征
        # 使用nn.Parameter，对weight和bias赋值
        self.fc1.weight = nn.Parameter(
                          torch.tensor([[0.1, 0.2],
                                        [0.3, 0.4],
                                        [0.5, 0.6]]))
        self.fc1.bias = nn.Parameter(
                          torch.tensor([0.1, 0.2, 0.3]))
        self.fc2 = nn.Linear(3, 2) # 定义第2个线性层fc2
        self.fc2.weight = nn.Parameter(  # 对线性层中的参数赋值
                          torch.tensor([[0.1, 0.2, 0.3],
                                        [0.4, 0.5, 0.6]]))
        self.fc2.bias = nn.Parameter(  # 对线性层中的参数赋值
                          torch.tensor([0.1, 0.2]))
        self.fc3 = nn.Linear(2, 2) # 定义第3个线性层fc3
        self.fc3.weight = nn.Parameter(  # 对线性层中的参数赋值
                          torch.tensor([[0.1, 0.2],
                                        [0.3, 0.4]]))
        self.fc3.bias = nn.Parameter(  # 对线性层中的参数赋值
                          torch.tensor([0.1, 0.2]))


    # 在forward函数中，实现神经网络的前向传播
    def forward(self, x): # 函数传入输入数据x
        z2 = self.fc1(x) # 将x输入到fc1中计算，得到线性结果z2
        a2 = torch.sigmoid(z2) # 使用sigmoid激活，得到a2
        z3 = self.fc2(a2)  # 将a2输入至fc2，得到z3
        a3 = torch.sigmoid(z3)  # z3输入至sigmoid，得到a3
        y = self.fc3(a3) #将a3输入至fc3，计算神经网络的输出y
        return y # 函数返回y


if __name__ == '__main__':
    # 设置3乘2的矩阵X，保存3个样本，每个样本2个特征
    X = torch.tensor([[1.0, 0.5],
                      [2.0, 3.0],
                      [5.0, 6.0]])
    network = Network()  # 定义神经网络模型network
    y = network(X) # 将X输入至network，计算输出y
    print(y) #打印y



















