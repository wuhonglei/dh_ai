"""
使用 vgg16 预训练模型进行图片分类
"""

import torch
import torch.nn as nn
from torchinfo import summary
import timm
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights, EfficientNet


class ImageModel(nn.Module):
    def __init__(self, num_classes: int, model_name: str, drop_rate: float):
        super(ImageModel, self).__init__()
        self.image_model = efficientnet_b7(
            weights=EfficientNet_B7_Weights.DEFAULT)
        self.image_model.classifier[-1] = nn.Linear(
            self.image_model.classifier[-1].in_features, num_classes)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.image_model(x)
        return x


if __name__ == '__main__':
    model = ImageModel(
        num_classes=10, model_name='efficientnet_b7', drop_rate=0.5)
    print(model)
    summary(model, input_size=(1, 3, 600, 600), col_names=[
            "input_size", "output_size", "num_params", "params_percent"])
