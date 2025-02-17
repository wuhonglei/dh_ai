""" 使用 vgg16 模型进行训练，识别 MNIST 手写数字 10 个类别 """

import os
import torch
import torch.nn as nn
from torchvision import models, transforms

from torchinfo import summary


# 设置 TORCH_HOME 环境变量
if os.path.exists('/mnt/model/nlp/'):
    os.environ['TORCH_HOME'] = '/mnt/model/nlp/pytorch'


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.feature = vgg.features.requires_grad_(False)
        self.classifier = vgg.classifier.requires_grad_(False)
        self.classifier[-1] = nn.Linear(in_features=4096,
                                        out_features=10).requires_grad_(True)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            prob = torch.softmax(x, 1)
            max_index = torch.argmax(prob, 1)
            return torch.argmax(x, 1), prob[0][max_index], prob[0]
