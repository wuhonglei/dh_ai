import torch
from torch import nn

# 定义神经网络类Network，它继承nn.Module类
class Network(nn.Module):
    # 神经网络中的神经元的数量是固定的，所以init不用传入参数
    def __init__(self):
        super().__init__() # 调用了父类的初始化函数
        # layer1是输入层与隐藏层之间的线性层
        self.layer1 = nn.Linear(28 * 28, 256)
        # layer2是隐藏层与输出层之间的线性层
        self.layer2 = nn.Linear(256, 10)

    # 实现神经网络的前向传播，函数传入输入数据x
    def forward(self, x):
        # 使用view函数将n*28*28的x张量，转换为n*784的张量，
        # 从而使得x可以被全连接层layer1计算
        x = x.view(-1, 28 * 28)
        x = self.layer1(x)  # 计算layer1的结果
        x = torch.relu(x)  # 进行relu激活
        # 计算layer2的结果，并返回
        return self.layer2(x)

#手动的遍历模型中的各个结构，并计算可以训练的参数
def print_parameters(model):
    cnt = 0
    for name, layer in model.named_children(): #遍历每一层
        # 打印层的名称和该层中包含的可训练参数
        print(f"layer({name}) parameters:")
        for p in layer.parameters():
            print(f'\t {p.shape} has {p.numel()} parameters')
            cnt += p.numel() #将参数数量累加至cnt
    #最后打印模型总参数数量
    print('The model has %d trainable parameters\n' % (cnt))

#打印输入张量x经过每一层时的维度变化情况
def print_forward(model, x):
    print(f"x: {x.shape}") # x从一个5*28*28的输入张量
    x = x.view(-1, 28 * 28) # 经过view函数，变成了一个5*784的张量
    print(f"after view: {x.shape}")
    x = model.layer1(x) #经过第1个线性层，得到5*256的张量
    print(f"after layer1: {x.shape}")
    x = torch.relu(x) #经过relu函数，没有变化
    print(f"after relu: {x.shape}")
    x = model.layer2(x) #经过第2个线性层，得到一个5*10的结果
    print(f"after layer2: {x.shape}")

if __name__ == '__main__':
    model = Network() #定义一个Network模型
    print(model) #将其打印，观察打印结果可以了解模型的结构
    print("")

    print_parameters(model) #将模型的参数打印出来

    #打印输入张量x经过每一层维度的变化情况
    x = torch.zeros([5, 28, 28])
    print_forward(model, x)

