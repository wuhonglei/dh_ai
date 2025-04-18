"""
使用多模态模型进行分类
text 使用 distilbert-base-uncased 模型
image 使用 resnet101 模型
"""

import torch
import torch.nn as nn
from torchinfo import summary
from transformers import AutoModel
import timm


class TextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super(TextEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids, attention_mask)
        x = outputs.last_hidden_state[:, 0, :]  # Get [CLS] token embedding
        return x


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str):
        super(ImageEncoder, self).__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0)  # Remove classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class MultiModalModel(nn.Module):
    def __init__(self, text_model_name: str, image_model_name: str, num_classes: int, drop_rate: float):
        super(MultiModalModel, self).__init__()
        self.text_encoder = TextEncoder(text_model_name)
        self.image_encoder = ImageEncoder(image_model_name)

        # Get feature dimensions
        text_features = self.text_encoder.model.config.hidden_size
        image_features = self.image_encoder.model.num_features

        self.dropout = nn.Dropout(drop_rate)

        self.image_classifier = nn.Linear(image_features, num_classes)
        self.text_classifier = nn.Linear(text_features, num_classes)
        self.fusion_weight = nn.Parameter(torch.ones(2))  # 可学习的权重

    def forward(self, text_input: torch.Tensor, attention_mask: torch.Tensor, image_input: torch.Tensor) -> torch.Tensor:
        text_features = self.text_encoder(text_input, attention_mask)
        image_features = self.image_encoder(image_input)

        image_output = self.image_classifier(self.dropout(image_features))
        text_output = self.text_classifier(self.dropout(text_features))

        fusion_output = self.fusion_weight[0] * \
            image_output + self.fusion_weight[1] * text_output

        return fusion_output


if __name__ == '__main__':
    model = MultiModalModel(
        text_model_name='distilbert-base-uncased',
        image_model_name='resnet101',
        num_classes=30,
        drop_rate=0.5
    )
    batch_size = 2
    seq_len = 30
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    image_input = torch.randn(batch_size, 3, 224, 224)
    summary(model, input_data=(input_ids, attention_mask, image_input))
