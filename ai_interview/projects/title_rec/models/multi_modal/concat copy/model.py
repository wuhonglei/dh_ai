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
    def __init__(self, text_model_name: str, image_model_name: str, num_classes: int, drop_rate: float, freeze_text_encoder: bool = True, freeze_image_encoder: bool = True):
        super(MultiModalModel, self).__init__()
        self.text_encoder = TextEncoder(text_model_name)
        self.image_encoder = ImageEncoder(image_model_name)

        # 不进行梯度更新
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # 不进行梯度更新
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        # Get feature dimensions
        text_features = self.text_encoder.model.config.hidden_size
        image_features = self.image_encoder.model.num_features

        # Unified classifier
        self.classifier = nn.Sequential(
            nn.Linear(text_features + image_features, 512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, text_input: torch.Tensor, attention_mask: torch.Tensor, image_input: torch.Tensor) -> torch.Tensor:
        text_features = self.text_encoder(text_input, attention_mask)
        image_features = self.image_encoder(image_input)

        # Concatenate features
        combined_features = torch.cat([text_features, image_features], dim=1)

        # Final classification
        output = self.classifier(combined_features)
        return output


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
