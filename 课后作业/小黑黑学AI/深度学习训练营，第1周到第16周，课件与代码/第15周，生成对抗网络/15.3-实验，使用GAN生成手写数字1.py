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

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import os
from torchvision import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def get_label(batch_size, label):
    return torch.full((batch_size, 1),
                      label,
                      dtype = torch.float,
                      device = device)

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data',
                             train = True,
                             download = True,
                             transform = transform)

    dataloader = DataLoader(dataset,
                            batch_size = 64,
                            shuffle = True)

    noise_size = 100  # size of the latent z vector
    feature_num = 256
    img_size = 28 * 28  # 图像大小为28x28

    netG = Generator(noise_size, feature_num, img_size).to(device)
    netD = Discriminator(img_size, feature_num, 1).to(device)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002)
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002)

    fixed_noise = torch.randn(64, noise_size, device=device)

    outf = "./gan-digit-result"
    try:
        os.makedirs(outf)
    except OSError:
        pass

    n_epoch = 200
    for epoch in range(n_epoch):
        for i, (data, _) in enumerate(dataloader):

            data = data.to(device)

            batch_size = data.size(0)

            netD.zero_grad()

            output = netD(data)
            D_x = output.mean().item()
            label = get_label(batch_size, 1)
            errD_real = criterion(output, label)

            noise1 = torch.randn(batch_size, noise_size, device=device)
            fake1 = netG(noise1)
            output = netD(fake1)
            D_G_z1 = output.mean().item()
            label = get_label(batch_size, 0)
            errD_fake = criterion(output, label)

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            #------------------
            netG.zero_grad()

            noise2 = torch.randn(batch_size, noise_size, device=device)
            fake2 = netG(noise2)
            output = netD(fake2)
            D_G_z2 = output.mean().item()
            label = get_label(batch_size, 1)

            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, n_epoch, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                utils.save_image(data,
                                 '%s/real_samples.png' % outf,
                                 normalize=True)
                fake = netG(fixed_noise)
                fake = fake.view(fake.size(0), 1, 28, 28)
                utils.save_image(fake.detach(),
                                 '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                                 normalize=True)

        torch.save(netG.cpu().state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
        torch.save(netD.cpu().state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
        netG.to(device)
        netD.to(device)

