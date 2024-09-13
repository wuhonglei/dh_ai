from torch import nn
import torch

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':
    noise_size = 100
    feature_num = 256
    img_size = 28 * 28

    netG = Generator(noise_size, feature_num, img_size).to(device)
    netG.load_state_dict(torch.load('digit.pth'))
    netG.eval()

    fixed_noise = torch.randn(64, noise_size, device=device)
    fake = netG(fixed_noise)

    # 设置子图网格的大小
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    axes = axes.flatten()  # 将网格展平，以便可以通过索引访问每个子图

    for i in range(64):
        # 重塑和缩放图像
        image = fake[i].detach().cpu().view(28, 28)  # 重塑为 28x28
        image = (image + 1) / 2  # 将[-1, 1]范围的图像缩放到[0, 1]

        # 显示图像
        axes[i].imshow(image, cmap='gray')  # 用灰度色彩图显示
        axes[i].axis('off')  # 关闭坐标轴

    plt.show()

