import torch
from torch import nn

# 定义判别器
class Discriminator(nn.Module):
    # 初始化函数init，传入输入层、隐藏层和输出层的神经元个数
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        # 定义了3个线性层fc1、fc2、fc3
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    # 实现神经网络的前向传播函数forward，函数传入张量x
    def forward(self, x):
        # 使用view函数，将x转为一维的向量
        x = x.view(x.size(0), -1)
        # x会经过fc1、fc2、fc3三个线性层
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        # 输入至sigmoid函数，得到生成器的输出
        return torch.sigmoid(x)

if __name__ == '__main__':
    img_size = 28 * 28 #设置图片维度，对应输入神经元的数量
    hidden_size = 256 #设置隐层藏神经元数量
    output_size = 1 #输出神经元，对应分类结果
    # 定义判别器netD
    netD = Discriminator(img_size, hidden_size, output_size)
    img = torch.randn(5, 28, 28) #定义5个随机的28*28的张量
    print(f"image: {img.shape}")
    output = netD(img) #输入至netD，得到结果output
    print(f"output: {output.shape}")

