import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        # 预训练的 VGG19 模型
        vgg_pretrained = models.vgg19(
            weights=models.VGG19_Weights.DEFAULT).features

        # 定义需要提取的层
        self.content_layers = ['21']  # conv4_2
        # conv1_1, conv2_1, etc.
        self.style_layers = ['0', '5', '10', '19', '28']

        self.model = nn.Sequential()
        self.layers = self.content_layers + self.style_layers
        self.layer_names = []  # 保存需要的层的名字(升序)

        # 组合需要的层
        for name, layer in vgg_pretrained._modules.items():
            self.model.add_module(name, layer)
            if name in self.layers:
                self.layer_names.append(name)

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layer_names:
                features[name] = x
        return features


def calc_content_loss(generated_features, content_features, weight):
    content_loss = F.mse_loss(
        generated_features['21'], content_features['21']
    )
    return content_loss * weight


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram
