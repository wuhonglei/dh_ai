import torch
import torch.nn as nn
import timm
from torchinfo import summary


class ImageEncoder(nn.Module):
    def __init__(self, model_name, pretrained: bool, trainable: bool):
        super().__init__()
        self.model = timm.create_model(  # type: ignore
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)
        return x


if __name__ == "__main__":
    image_encoder = ImageEncoder(
        model_name="resnet50", pretrained=True, trainable=True
    )
    print(image_encoder)
    summary(image_encoder, input_size=(1, 3, 224, 224))
