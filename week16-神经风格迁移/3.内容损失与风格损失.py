import torch
import torch.nn as nn
import torchvision.models as models


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
        idx = 0

        # 组合需要的层
        for name, layer in vgg_pretrained._modules.items():
            self.model.add_module(name, layer)
            if name in self.layers:
                self.layer_names.append(name)
            idx += 1

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layer_names:
                features[name] = x
        return features


vgg = VGGFeatures()
print(vgg.idx)
