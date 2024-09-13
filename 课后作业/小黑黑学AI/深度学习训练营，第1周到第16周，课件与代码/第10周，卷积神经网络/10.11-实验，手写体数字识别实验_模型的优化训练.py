
import torch
from torch.nn import Module
from torch import nn

class LeNet5(Module): #继承torch.nn中的Module模块
    def __init__(self): #定义LeNet5模型
        super(LeNet5, self).__init__()
        # 有1个输入通道，64个输出通道，卷积核的大小为5乘5
        self.layer1 = nn.Sequential(
                    nn.Conv2d(1, 64, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout(0.25)) #增加Dropout
        #有64个输入通道，256个输出通道，卷积核的大小为5乘5
        self.layer2 = nn.Sequential(
                    nn.Conv2d(64, 256, 5),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout(0.25))
        # 输入神经元个数是4096 = 256 * 4 * 4，输出神经元1024
        self.fc1 = nn.Sequential(
                    nn.Linear(4096, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.25))
        # 输入神经元个数是4096 = 256 * 4 * 4，输出神经元1024
        self.fc2 = nn.Sequential(
                    nn.Linear(1024, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25))
        #第3个全连接层fc3，是最后的输出层
        #有256个输入，10个输出，这10个输出分别对应数字0到9
        self.fc3 = nn.Linear(256, 10)

    #函数输入一个四维张量x
    #这四个维度分别是样本数量、输入通道、图片的高度和宽度
    def forward(self, x): #[n, 1, 28, 28]
        y = self.layer1(x) #[n, 64, 24, 24]
        y = self.layer2(y) #[n, 256, 4, 4]
        #使用view函数，将张量的维度从n乘256乘4乘4转为n乘4096
        y = y.view(y.shape[0], -1) #[n, 4096]
        y = self.fc1(y)  # [n, 1024]
        y = self.fc2(y)  # [n, 256]
        y = self.fc3(y)  # [n, 10]
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
    #设置device保存当前的设备，判断cuda是否可以使用
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    #接着将模型与损失函数的张量转为该模式
    #也就是如果当前可以使用gpu，则将张量转为gpu的模式
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
    torch.save(model.state_dict(), 'lenet5.update.pt')

