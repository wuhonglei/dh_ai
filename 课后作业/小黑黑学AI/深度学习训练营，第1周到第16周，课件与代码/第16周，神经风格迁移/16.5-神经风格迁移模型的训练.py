import torch
import torch.nn as nn
import torch.nn.functional as F

# 计算新生成图像在内容方面的损失值
class ContentLoss(nn.Module):
    def __init__(self, target):
        # 将原内容图像的特征图作为target输入
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        # 计算新图像input与target的均方误差
        self.loss = F.mse_loss(input, self.target)
        return input

# 格拉姆矩阵用于度量特征图中的不同通道之间的相关性
# 计算特征图target的格拉姆矩阵
def gram_matrix(target):
    # 四个维度的含义分别是批量大小、通道数、特征图的高度和宽度
    b, c, h, w = target.size() #获取target四个通道的维度
    # 重塑张量，将原4维的target，转为2维的矩阵
    # 矩阵的行是b*c，列是h*w
    features = target.view(b * c, h * w)
    # 计算feature乘feature的转置，结果代表了不同特征图之间的相关性
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w) #使用div对G进行规范化


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        # 计算输入特征图target的格拉姆矩阵
        self.target = gram_matrix(target).detach()
    def forward(self, input):
        G = gram_matrix(input)
        # 计算两个格拉姆矩阵的MSE
        self.loss = F.mse_loss(G, self.target)
        return input

# 标准化层
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach()
        self.std = std.clone().detach()
    def forward(self, img):
        return (img - self.mean) / self.std

# 函数传入VGG-19网络cnn和样式图片style_img和内容图片content_img
def create_model_and_losses(cnn, style_img, content_img):
    normal_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    normal_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    # 构造一个标准化层normal
    # 标准化层可以加快收敛速度，并防止梯度消失和梯度爆炸等问题
    normal = Normalization(normal_mean, normal_std)
    model = nn.Sequential(normal) #将normal添加到新的模型model中
    content_layers = {'conv_4'} # 定义内容层名称集合
    # 定义样式层的名称集合
    style_layers = {'conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'}
    content_losses = [] #保存内容损失的计算方法
    style_losses = [] #保存样式损失的计算方法

    cnt = 0
    # 遍历VGG-19的各层，为各层添加名称conv、relu、pool等等
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            cnt += 1
            name = 'conv_{}'.format(cnt)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(cnt)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(cnt)
        else:
            name = 'other_{}'.format(cnt)
        model.add_module(name, layer) #将这些层添加到model中

        # 如果正在遍历的层在content_layers中
        if name in content_layers:
            # 将内容图片content_img输入至这个网络，提前计算出这个位置的特征图target
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            # 将计算生成图片内容损失的方法ContentLoss添加到model
            # 后续即可直接与ContentLoss中的原始内容特征图，计算MSE均方误差
            model.add_module("content_loss_{}".format(cnt), content_loss)
            content_losses.append(content_loss)

        # StyleLoss的计算方式也是同样的道理
        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module("style_loss_{}".format(cnt), style_loss)
            style_losses.append(style_loss)

    i = 0
    # 将无关的层截取掉，来提升计算的速度
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) \
        or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses # 函数返回

from PIL import Image
from torchvision import transforms
# 读取图片的函数
def image_loader(image_name, target_size):
    image = Image.open(image_name)
    image = image.resize(target_size) # 重新修改图片的尺寸
    transform = transforms.ToTensor() # 将图片转为张量
    image = transform(image).unsqueeze(0) # 添加一个维度
    return image

# 将张量形式的图像image，转换为图像格式
def imsave(tensor, save_file):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    transform = transforms.ToPILImage()
    image = transform(image)
    image.save(save_file) # 将图像保存下来

import torchvision.models as models
from torchvision.models import VGG19_Weights
import torch.optim as optim
import os

if __name__ == "__main__":
    # 获取到内容图像的尺寸，保存到content_size中
    content_size = Image.open("./data/tower.jpg").size
    # 读取风格图像
    # 由于在内容图像上，进行风格变换，因此需要将样式图像的大小修改为内容图像的大小
    style_img = image_loader("./data/vango.jpg",
                             target_size = content_size)
    # 读取内容图像
    content_img = image_loader("./data/tower.jpg",
                               target_size = content_size)

    # 定义VGG-19网络
    cnn = models.vgg19(weights = VGG19_Weights.DEFAULT).features.eval()
    # 创建计算损失的模型model和损失函数style_losses与content_losses
    model, style_losses, content_losses = \
        create_model_and_losses(cnn, style_img, content_img)

    model.eval() # 将模型调整为评估模式
    model.requires_grad_(False) # 设置梯度为False

    # 定义生成图片，初始时，直接复制内容图片中的信息
    generate_img = content_img.clone()
    # 将该图片中的像素点的梯度设置为True，接下来我们要迭代图像中的每个像素点
    generate_img.requires_grad_(True)
    # 在迭代时，会使用LBFGS优化器
    # 在风格迁移问题上，LBFGS比Adam等其他优化器更加有效，并且有更好的迭代效率
    optimizer = optim.LBFGS([generate_img])

    # 创建generate_pic文件夹，用于保存迭代结果
    save_path = './generate_pic/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 接下来我们将迭代100轮，将每一轮的结果，都保存到文件夹中
    iterate_num = 100

    # 因为需要在闭包函数closure中使用这个变量，所以我们要将epoch，定义为数组
    epoch = [0] # 表示迭代轮数的变量

    while epoch[0] < iterate_num: # 进入循环迭代
        # 因为实际的迭代次数，是closure函数的调用次数
        # 因此要在closure函数中控制epoch的更新

        def closure(): # 将迭代的过程，在closure函数中定义
            with torch.no_grad():
                # 确保生成图像的像素值在0到1之间
                generate_img.clamp_(0, 1)
            optimizer.zero_grad() # 清空梯度

            # 使用model执行前向传播，计算风格损失和内容损失
            model(generate_img)
            style_loss = 0 # 风格损失
            content_loss = 0 # 内容损失
            # 将两种损失累加到style_loss和content_loss中
            for sl in style_losses:
                style_loss += sl.loss
            for cl in content_losses:
                content_loss += cl.loss

            # 将style_loss放大100万倍，增加风格特征的重要性
            style_loss = style_loss * 1000000
            loss = style_loss + content_loss #累加两种损失到loss中
            loss.backward() #调用backward计算梯度

            epoch[0] += 1
            # 打印调试信息
            print(f"Epoch {epoch[0]}: " # 迭代的轮数
                  f"Style Loss : {style_loss.item():4f}" # 风格损失
                  f" Content Loss: {content_loss.item():4f}") # 内容损失
            # 将这一轮迭代生成的图像，保存下来
            imsave(generate_img, save_path + f"epoch{epoch[0]}.jpg")
            return loss # 返回loss

        optimizer.step(closure) #进行迭代

