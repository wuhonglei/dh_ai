

import torch
from torch.nn import Module
from torch import nn

class LeNet5(Module): #继承torch.nn中的Module模块
    def __init__(self): #定义LeNet5模型
        super(LeNet5, self).__init__()
        #第1个卷积层conv1，它有1个输入通道，6个输出通道，卷积核的大小为5乘5
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU() #激活函数使用relu
        #池化层pool1，它是大小为2乘2的最大池化
        self.pool1 = nn.MaxPool2d(2)

        #第2个卷积层conv2，它有6个输入通道，16个输出通道，卷积核的大小为5乘5
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU() #激活函数使用relu
        self.pool2 = nn.MaxPool2d(2) #2乘2的最大池化

        #第1个全连接层的输入神经元有256个，设置120个输出神经
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU() #激活函数使用relu

        #第2个全连接层fc2是120乘84的，即120个输入，84个输出
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU() #激活函数使用relu

        #第3个全连接层fc3，是最后的输出层
        #有84个输入，10个输出，这10个输出分别对应数字0到9
        self.fc3 = nn.Linear(84, 10)

    #函数输入一个四维张量x
    #这四个维度分别是样本数量、输入通道、图片的高度和宽度
    def forward(self, x): #[n, 1, 28, 28]
        y = self.conv1(x) #[n, 6, 24, 24]
        y = self.relu1(y) #[n, 6, 24, 24]
        y = self.pool1(y) #[n, 6, 12, 12]
        y = self.conv2(y) #[n, 16, 8, 8]
        y = self.relu2(y) #[n, 16, 8, 8]
        y = self.pool2(y) #[n, 16, 4, 4]
        #使用view函数，将张量的维度从n乘16乘4乘4转为n乘256
        y = y.view(y.shape[0], -1) #[n, 256]
        y = self.fc1(y) #[n, 120]
        y = self.relu3(y) #[n, 120]
        y = self.fc2(y) #[n, 84]
        y = self.relu4(y) #[n, 84]
        y = self.fc3(y) #[n, 10]
        return y #函数输出一个n乘10的结果

from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

if __name__ == '__main__':
    #同样需要读取MNIST数据集
    #需要将mnist函数的train参数设置为False，表示读取测试数据
    test_data = mnist.MNIST(root='./data',
                            train=False,
                            download=True,
                            transform=ToTensor())
    #打印test_data的维度，是[10000, 28, 28]，即1万个28乘28的图片
    print(f"test data size: {test_data.data.shape}")
    #使用DataLoader读取test_data，不需要设置参数，这样会一个一个数据的读取
    test_loader = DataLoader(test_data)

    model = LeNet5()  # 创建一个Lenet5模型对象
    # 设置device保存当前的设备，判断cuda是否可以使用
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    # 接着将模型与损失函数的张量转为该模式
    # 也就是如果当前可以使用gpu，则将张量转为gpu的模式
    model = model.to(device)
    #读取刚刚训练的lenet5模型文件
    model.load_state_dict(torch.load('lenet5.pt'))
    model.eval() #将模型设置为测试模式

    right = 0 #设置right变量，保存预测正确的样本数量
    all = 0 #all保存全部的样本数量
    #遍历test_loader中的全部数据
    #其中x表示样本的特征张量，y表示样本的标签
    for (x, y) in test_loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x) #使用模型预测x的结果
        #检查y_pred和y是否相同
        if y_pred.argmax(1).eq(y)[0] == True:
            right += 1 #如果相同，那么right累加1
        all += 1 #每次循环all变量都加1
    #循环结束后，计算正确率，并打印结果
    acc = right * 1.0 / all
    print("test accuracy = %d / %d = %.3lf"%(right, all, acc))


