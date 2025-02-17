import os
import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary
from collections import namedtuple
from torch.nn.functional import mse_loss

# 设置 TORCH_HOME 环境变量
if os.path.exists('/mnt/model/nlp/'):
    os.environ['TORCH_HOME'] = '/mnt/model/nlp/pytorch'


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(0, 4):
            self.slice1.add_module(
                str(x), vgg_pretrained_features[x])  # type: ignore
        for x in range(4, 9):
            self.slice2.add_module(
                str(x), vgg_pretrained_features[x])  # type: ignore
        for x in range(9, 16):
            self.slice3.add_module(
                str(x), vgg_pretrained_features[x])  # type: ignore
        for x in range(16, 23):
            self.slice4.add_module(
                str(x), vgg_pretrained_features[x])  # type: ignore

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


if __name__ == '__main__':
    model = VGG16()
    print(model)
    summary(model, input_size=(1, 3, 224, 224), col_names=(
        "input_size", "output_size", "num_params"))
