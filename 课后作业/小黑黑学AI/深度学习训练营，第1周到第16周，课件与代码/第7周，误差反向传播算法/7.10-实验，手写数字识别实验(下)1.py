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

#从torchvision.datasets中导入MNIST模块，然后读入MNIST数据集
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim

if __name__ == '__main__':
    # 使用mnist函数导入数据，函数包括了4个参数
    # root代表数据所在的文件夹
    # train=True表示读入训练数据
    # download=True表示如果数据不在文件夹中，那么就从互联网中下载
    # transform=ToTensor()，表示将读入的数据转为张量的形式
    train_data = mnist.MNIST(root='./data',
                             train=True,
                             download=True,
                             transform=ToTensor())
    # MNIST数据集的训练集，一共包含了6万个数据
    # 打印train_data的维度
    print(f"traind_data size = {train_data.data.shape}")

    # 使用DataLoader对读取和处理数据
    # 设置参数shuffle=True，代表随机打乱数据的顺序
    # 参数batch_size=128，代表每次会读取128个样本，作为一组训练数据
    train_load = DataLoader(train_data,
                            shuffle=True,
                            batch_size=128)
    # 打印train_load的长度
    print(f"train_load size = {len(train_load)}")

    model = Network()  # 创建一个Network模型对象
    optimizer = optim.Adam(model.parameters())  # 创建一个Adam优化器
    criterion = nn.CrossEntropyLoss()  # 创建一个交叉熵损失函数
    
    
    # 进入模型的循环迭代
    for epoch in range(10): #外层循环，代表了整个训练数据集的遍历次数
        # 内层循环代表了，在一个epoch中，以批量的方式，使用train_load对于数据进行遍历
        # batch_idx 表示当前遍历的批次
        # (data, label) 表示这个批次的训练数据和标记。
        for batch_idx, (data, label) in enumerate(train_load):
            # 使用当前的模型，预测训练数据data，结果保存在output中
            output = model(data)

            # 调用criterion，计算预测值output与真实值label之间的损失loss
            loss = criterion(output, label)
            loss.backward()  # 计算损失函数关于模型参数的梯度
            optimizer.step()  # 更新模型参数
            optimizer.zero_grad() # 将梯度清零，以便于下一次迭代

            # 对于每个epoch，每训练100个batch，打印一次当前的损失
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/10 "
                      f"| Batch {batch_idx}/{len(train_load)} "
                      f"| Loss: {loss.item():.4f}")

    # 将训练好的模型保存为文件，文件名是network.digit
    torch.save(model.state_dict(), 'network.digit')









