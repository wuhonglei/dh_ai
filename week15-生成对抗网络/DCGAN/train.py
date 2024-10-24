"""
GAN 生成对抗网络训练代码
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


from generator import Generator
from discriminator import Discriminator
from dataset import AnimateDataset
from utils import make_dirs


def main():
    epochs = 300
    batch_size = 1024
    img_size = 64
    n_channels = 3  # 图像通道数, RGB
    input_size = 100  # 噪声维度
    output_dir = './output'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_noise = torch.randn(64, input_size).to(device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = AnimateDataset(root='./images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=3, pin_memory=True)

    # 创建生成器和判别器

    generator = Generator(input_size, n_channels).to(device)
    discriminator = Discriminator(n_channels).to(device)

    # 定义损失函数和优化器
    g_optimizer = torch.optim.Adam(
        generator.parameters(), lr=0.0002)  # 生成器的优化器
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=0.0002)  # 判别器的优化器
    criterion = nn.BCELoss()  # 二分类交叉熵损失函数

    make_dirs(output_dir, remove=True)
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            """ 训练判别器 """
            d_optimizer.zero_grad()
            output_d = discriminator(real_imgs)
            loss_d_real = criterion(output_d, torch.ones_like(output_d))

            noise = torch.randn(batch_size, input_size).to(device)
            fake_imgs = generator(noise)
            output_g = discriminator(fake_imgs)
            loss_d_fake = criterion(output_g, torch.zeros_like(output_g))
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            d_optimizer.step()

            """ 训练生成器 """
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            noise = torch.randn(batch_size, input_size).to(device)
            fake_imgs = generator(noise)
            output = discriminator(fake_imgs)
            loss_g = criterion(output, torch.ones_like(output))
            loss_g.backward()
            g_optimizer.step()

            print(f'Epoch [{epoch}/{epochs}], Step [{i}/{len(dataloader)}], '
                  f'Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}')

        fake_imgs = generator(fixed_noise)
        fake_imgs = fake_imgs.view(fake_imgs.size(0), 3, 64, 64)
        # 保存生成的图像
        img_path = os.path.join(
            output_dir, f'fake_images-{epoch+1:03d}.png')
        save_image(
            fake_imgs.detach(), img_path, nrow=8, normalize=True)

        if (epoch + 1) % 10 == 0:
            dist = f'./models/{epoch+1}'
            make_dirs(dist, remove=True)
            torch.save(generator.state_dict(),
                       f'{dist}/generator.pth')
            torch.save(discriminator.state_dict(),
                       f'{dist}/discriminator.pth')


def random_generate():
    fixed_noise = torch.randn(64, 100)
    generator = Generator(100, 3)
    # generator.load_state_dict(torch.load(
    #     './models/100/generator.pth', map_location='cpu', weights_only=True))

    fake = generator(fixed_noise)
    image = fake.detach().view(-1, 3, 64, 64)
    save_image(image, 'fake_images.png', nrow=8, normalize=True)


if __name__ == '__main__':
    main()
    # random_generate()
