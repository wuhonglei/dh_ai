"""
使用 vgg16 预训练模型进行图片分类
"""

import torch
import torch.nn as nn
from torchinfo import summary
import timm


class ImageModel(nn.Module):
    def __init__(self, num_classes: int, model_name: str, drop_rate: float):
        super(ImageModel, self).__init__()
        self.vgg = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            in_chans=3,
            drop_rate=drop_rate,
            global_pool='avg'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vgg(x)
        return x


if __name__ == '__main__':
    model = ImageModel(num_classes=10, model_name='vgg16', drop_rate=0.5)
    print(model)
    summary(model, input_size=(1, 3, 224, 224))
