
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

#从torchvision.datasets中导入MNIST模块，然后将数据直接导入
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim

if __name__ == '__main__':
    #导入数据的mnist函数包括了4个参数
    #root代表数据所在的文件夹
    #train=True表示训练数据
    #download=True表示如果数据不再对应文件夹中，则从互联网中下载
    #transform=ToTensor()表示将数据转为张量
    train_data = mnist.MNIST(root='./data',
                             train=True,
                             download=True,
                             transform=ToTensor())
    #torch.Size([60000, 28, 28])
    #print(train_data.data.shape)
    #使用DataLoader对数据进行读取，并设置shuffle=True，随机的打乱数据顺序
    #batch_size=128，每次会读取128个数据作为一组进行训练
    train_load = DataLoader(train_data, shuffle=True, batch_size=128)

    model = LeNet5() #创建一个Lenet5模型对象
    optimizer = optim.Adam(model.parameters()) #创建一个Adam优化器
    criterion = nn.CrossEntropyLoss() #创建一个交叉熵损失函数

    #在pytorch训练时，可以选择cpu或gpu两种方式进行训练
    #如果当前机器的cuda可以使用，那么就使用gpu模式，否则使用cpu
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    #device会等于cuda或cpu，将代码的中使用的张量都转为该模式
    #将模型与损失函数的张量转为device保存的模式
    model = model.to(device)
    criterion = criterion.to(device)

    epochs = 10000 #设置总的迭代次数epochs等于10000
    for i in range(1, epochs + 1): #进入迭代的循环
        model.train() #设置模型为训练模式
        #从train_load获取一组训练数据
        #样本的特征向量保存到x中，标签保存到y中
        x, y = next(iter(train_load))
        #x表示，每组128个数据，1个输入通道，图片大小为28乘28
        #y的长度是128，代表128个样本的标签，每个标签的取值为0到9
        #print(x.shape) #torch.Size([128, 1, 28, 28])
        #print(y.shape) #torch.Size([128])
        x = x.to(device) #将x转为当前的设备模式
        y = y.to(device) #将y转为当前的设备模式

        #将模型中参数的梯度置为0，准备新的一轮训练
        optimizer.zero_grad()
        #使用当前的模型model预测张量x，预测的输出保存到y_pred中
        y_pred = model(x)
        #接着将预测值和真实值传入损失函数中，计算损失
        loss = criterion(y_pred, y)
        loss.backward() #将损失向输入侧进行反向传播，计算梯度
        optimizer.step() #使用优化器对x的值进行更新

        if i % 100 == 0: #每迭代100轮
            #计算一次当前一轮训练的正确率
            acc = y_pred.argmax(1).eq(y).sum().float() / 128
            print("iterate %d: acc = %.3lf"%(i, acc.item()))

    #循环迭代结束后，将模型保存
    torch.save(model.state_dict(), 'lenet5.pt')

