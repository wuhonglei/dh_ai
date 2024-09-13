from PIL import Image
import os

# 设置CaptchaDataset继承Dataset，用于读取验证码数据
from torch.utils.data import Dataset
# 设置AnimeDataset继承Dataset，用于读取名字训练数据
class AnimeDataset(Dataset):
    # init函数用于初始化
    # init函数用于初始化，函数传入数据的路径data_dir
    def __init__(self, data_dir, transform):
        self.file_list = list()  # 保存每个训练数据的路径
        # 使用os.listdir，获取data_dir中的全部文件
        files = os.listdir(data_dir)
        for file in files:  # 遍历files
            # 将目录路径与文件名组合为文件路径
            path = os.path.join(data_dir, file)
            # 将path添加到file_list列表
            self.file_list.append(path)
        # 将数据转换对象transform保存到类中
        self.transform = transform
        self.length = len(self.file_list)  # 保存数据的个数

    def __len__(self):
        # 直接返回数据集中的样本数量
        # 重写该方法后可以使用len(dataset)语法，来获取数据集的大小
        return self.length

    # 函数getitem传入索引index
    def __getitem__(self, index):
        file_path = self.file_list[index] #获取数据的路径
        image = Image.open(file_path)
        image = self.transform(image)
        return image

from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        self.ct1 = nn.ConvTranspose2d(
                        in_channels = noise_size,
                        out_channels = 1024,
                        kernel_size = 4,
                        stride = 1,
                        padding = 0,
                        bias = False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()

        self.ct2 = nn.ConvTranspose2d(
                        in_channels = 1024,
                        out_channels = 512,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)

        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU()

        self.ct3 = nn.ConvTranspose2d(
                        in_channels = 512,
                        out_channels = 256,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.ct4 = nn.ConvTranspose2d(
                        in_channels = 256,
                        out_channels = 128,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        self.ct5 = nn.ConvTranspose2d(
                        in_channels = 128,
                        out_channels = 3,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.ct1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.ct2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.ct3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.ct4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.ct5(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
                        in_channels = 3,
                        out_channels = 64,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.leaky_relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(
                        in_channels = 64,
                        out_channels = 128,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.bn2 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(
                        in_channels = 128,
                        out_channels = 256,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1,
                        bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        self.leaky_relu3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(
                        in_channels = 256,
                        out_channels = 512,
                        kernel_size = 4,
                        stride = 2,
                        padding= 1,
                        bias = False)
        self.bn4 = nn.BatchNorm2d(512)
        self.leaky_relu4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(
                        in_channels = 512,
                        out_channels = 1,
                        kernel_size = 4,
                        stride = 1,
                        padding = 0,
                        bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu4(x)

        x = self.conv5(x)
        x = self.sigmoid(x)
        return x.view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
import torch
from torchvision import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def get_label(batch_size, label):
    return torch.full((batch_size,),
                      label,
                      dtype=torch.float,
                      device=device)

if __name__ == '__main__':
    # 定义数据转换对象transform
    # 使用transforms.Compose定义数据预处理流水线
    # 在transform添加Resize和ToTensor两个数据处理操作
    trans = transforms.Compose([transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

    # 定义CaptchaDataset对象dataset
    dataset = AnimeDataset('./anime-face', trans)
    # 定义数据加载器data_load
    # 其中参数dataset是数据集
    # batch_size=8代表每个小批量数据的大小是8
    # shuffle = True表示每个epoch，都会随机打乱数据的顺序
    dataloader = DataLoader(dataset,
                            batch_size=64,
                            shuffle=True)

    noise_size = 100  # size of the latent z vector
    netG = Generator(noise_size).to(device)
    netG.apply(weights_init)
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(),
                            lr=0.0002,
                            betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(),
                            lr=0.0002,
                            betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, noise_size, 1, 1, device=device)

    outf = "./anime-face-result"
    try:
        os.makedirs(outf)
    except OSError:
        pass
    niter = 200
    for epoch in range(niter):
        for i, data in enumerate(dataloader):
            data = data.to(device)
            batch_size = data.size(0)

            netD.zero_grad()

            output = netD(data)
            D_x = output.mean().item()
            label = get_label(batch_size, 1)
            errD_real = criterion(output, label)

            noise1 = torch.randn(batch_size, noise_size, 1, 1, device=device)
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

            noise2 = torch.randn(batch_size, noise_size, 1, 1, device=device)
            fake2 = netG(noise2)
            output = netD(fake2)
            D_G_z2 = output.mean().item()
            label = get_label(batch_size, 1)

            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if i % 100 == 0:
                utils.save_image(data,
                                 '%s/real_samples.png' % outf,
                                 normalize=True)
                fake = netG(fixed_noise)
                utils.save_image(fake.detach(),
                                 '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                                 normalize=True)

        torch.save(netG.cpu().state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
        torch.save(netD.cpu().state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
        netG.to(device)
        netD.to(device)
