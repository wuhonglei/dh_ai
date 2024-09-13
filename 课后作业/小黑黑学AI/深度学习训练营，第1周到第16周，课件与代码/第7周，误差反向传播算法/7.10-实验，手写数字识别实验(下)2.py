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

if __name__ == '__main__':
    # 同样需要读取MNIST数据集
    # 将mnist函数的train参数设置为False，表示读取测试数据
    test_data = mnist.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=ToTensor())
    # MNIST测试数据集一共有1万个
    # 打印test_data的维度
    print(f"test data size: {test_data.data.shape}")
    # 使用DataLoader读取test_data
    # 不需要设置任何参数，这样会一个一个数据的读取
    test_loader = DataLoader(test_data)


    model = Network()  # 创建一个network模型
    # 调用load_state_dict，读取已经训练好的模型文件network.digit
    model.load_state_dict(torch.load('network.digit'))

    right = 0  # 设置right变量，保存预测正确的样本数量
    all = 0  # all保存全部的样本数量
    # 遍历test_loader中的数据
    # x表示样本的特征张量，y表示样本的标签
    for (x, y) in test_loader:
        pred = model(x)  # 使用模型预测x的结果，保存在pred中
        # 检查pred和y是否相同
        if pred.argmax(1).eq(y)[0] == True:
            right += 1  # 如果相同，那么right加1
        all += 1  # 每次循环，all变量加1

    # 循环结束后，计算模型的正确率
    acc = right * 1.0 / all
    print("test accuracy = %d / %d = %.3lf" % (right, all, acc))


