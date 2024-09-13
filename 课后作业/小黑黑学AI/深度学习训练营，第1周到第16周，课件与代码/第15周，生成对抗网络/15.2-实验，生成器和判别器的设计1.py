import torch
from torch import nn

# 定义生成器神经网络
class Generator(nn.Module):
    # 网络的初始化函数init，传入输入层、隐藏层和输出层的神经元个数
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        # 定义了3个线性层fc1、fc2、fc3
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    # 神经网络的前向传播函数forward，函数传入张量x
    def forward(self, x):
        # x会经过fc1、fc2、fc3三个线性层
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        # 生成器的最后一个激活函数，一般会使用双曲正切函数tanh
        # tanh的输出范围在[-1, 1]，可以使生成的图像数据
        # 更容易匹配到真实图像数据的分布
        x = torch.tanh(x)
        return x

import matplotlib.pyplot as plt

if __name__ == '__main__':
    noise_size = 100 #设置噪声向量的维度，对应输入神经元的数量
    hidden_size = 256 #设置隐层藏神经元数量
    img_size = 28 * 28 #输出图像的大小，对应输出神经元的数量

    netG = Generator(noise_size, hidden_size, img_size) #定义生成器
    noise = torch.randn(3, noise_size) #噪声向量，对应3个样本
    print(f'fixed_noise shape:{noise.shape}')
    # 将噪声noise输入到netG，就得到了生成器输出的假图张量fake
    fake = netG(noise)
    print(f'fake shape:{fake.shape}')
    # 将fake转为3个28×28的图像数据image
    image = fake.detach().view(-1, 28, 28)
    print(f'fake shape:{image.shape}')

    # 将它们画在画板上
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for i, ax in enumerate(axes):
        ax.imshow(image[i], cmap='gray')
        ax.axis('off')
    plt.show()
