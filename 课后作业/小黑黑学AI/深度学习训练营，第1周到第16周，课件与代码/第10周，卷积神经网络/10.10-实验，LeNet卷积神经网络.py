import torch
from torch.nn import Module
from torch import nn

class LeNet5(Module): #继承torch.nn中的Module模块
    def __init__(self): #定义LeNet5模型
        super(LeNet5, self).__init__()
        #第1个卷积层conv1，它有1个输入通道，6个输出通道，卷积核的大小为5乘5
        #该卷积核张量的维度是[6, 1, 5, 5]，包含了150个w参数，6个b参数
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU() #激活函数使用relu
        #池化层pool1，它是大小为2乘2的最大池化
        self.pool1 = nn.MaxPool2d(2)

        #经过conv1，输入数据的变化
        #输入: [n, 1, 28, 28]
        #卷积核: 5×5
        #卷积后: [n, 6, 24, 24]
        #池化后: [n, 6, 12, 12]

        #第2个卷积层conv2，它有6个输入通道，16个输出通道，卷积核的大小为5乘5
        #卷积核张量的维度是[16, 6, 5, 5]，包含了2400个w参数，16个b参数
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU() #激活函数使用relu
        self.pool2 = nn.MaxPool2d(2) #2乘2的最大池化

        # 经过conv2，输入数据的变化
        # 输入: [n, 6, 12, 12]
        # 卷积核: 5×5
        # 卷积后: [n, 16, 8, 8]
        # 池化后: [n, 16, 4, 4]

        #第1个全连接层的输入神经元有256个，设置120个输出神经
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU() #激活函数使用relu
        #该层中包含的参数数量: 256*120=30720个w参数，120个b参数

        #第2个全连接层fc2是120乘84的，即120个输入，84个输出
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU() #激活函数使用relu
        # 该层中包含的参数数量: 120*84=10080个w参数，84个b参数

        #第3个全连接层fc3，是最后的输出层
        #有84个输入，10个输出，这10个输出分别对应数字0到9
        self.fc3 = nn.Linear(84, 10)
        # 该层中包含的参数数量: 84*10=840个w参数，10个b参数


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
        return y #函数返回一个n乘10的结果



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

#打印输入张量x经过每一层维度的变化情况
def print_forward(model, x):
    print("input x to model:")
    print(f"x: {x.shape}")
    x = model.conv1(x)
    print(f"after conv1: {x.shape}")
    x = model.relu1(x)
    print(f"after relu1: {x.shape}")
    x = model.pool1(x)
    print(f"after pool1: {x.shape}")
    x = model.conv2(x)
    print(f"after conv2: {x.shape}")
    x = model.relu2(x)
    print(f"after relu2: {x.shape}")
    x = model.pool2(x)
    print(f"after pool2: {x.shape}")
    x = x.view(x.shape[0], -1)
    print(f"after view: {x.shape}")
    x = model.fc1(x)
    print(f"after fc1: {x.shape}")
    x = model.relu3(x)
    print(f"after relu3: {x.shape}")
    x = model.fc2(x)
    print(f"after fc2: {x.shape}")
    x = model.relu4(x)
    print(f"after relu4: {x.shape}")
    x = model.fc3(x)
    print(f"after fc3: {x.shape}")

if __name__ == '__main__':
    model = LeNet5() #定义一个LeNet5模型
    print(model) #将其打印，观察打印结果可以了解模型的结构
    print("")

    print_parameters(model) #将模型的参数打印出来

    #打印输入张量x经过每一层维度的变化情况
    x = torch.zeros([5, 1, 28, 28])
    print_forward(model, x)


