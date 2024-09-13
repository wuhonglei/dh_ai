import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from torchinfo import summary


def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert('RGB')
    size = min(max_size, max(image.size))
    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = in_transform(image).unsqueeze(0)  # 为了符合模型输入的格式，增加一个维度
    return image


class VGG19(nn.Module):
    """ 从VGG19中提取特征 """

    def __init__(self):
        super(VGG19, self).__init__()
        vgg_pretrained = models.vgg19(
            weights=models.VGG19_Weights.DEFAULT).features

        self.styled_layers = ['0', '5', '10', '19', '28']  # 五个卷积层的首位索引
        self.content_layer = '21'  # 21是VGG19中的第四个卷积层
        self.model = nn.Sequential()
        for name, layer in vgg_pretrained.named_children():
            self.model.add_module(name, layer)

    def forward(self, x):
        styles = []
        content = None
        for name, layer in self.model.named_children():
            x = layer(x)
            if name in self.styled_layers:
                styles.append(x)
            if name == self.content_layer:
                content = x
        return content, styles


def content_loss(content_feature, target):
    # 计算内容损失和风格损失
    return torch.mean((content_feature - target) ** 2)
