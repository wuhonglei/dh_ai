from modules.image_encoder import ImageEncoder
from modules.text_encoder import TextEncoder
import torch.nn as nn
import torch


class CLIPModel(nn.Module):
    def __init__(
        self,
            text_model_name,
            image_model_name,
            text_pretrained,
            text_trainable,
            image_pretrained,
            image_trainable,
            text_embedding_dim,
            image_embedding_dim,
            projection_dim,
    ):
        super(CLIPModel, self).__init__()
        self.image_encoder = ImageEncoder(
            image_model_name, image_pretrained, image_trainable)
        self.text_encoder = TextEncoder(
            text_model_name, text_pretrained, text_trainable)
        self.image_projection = nn.Parameter(
            torch.randn(image_embedding_dim, projection_dim))
        self.text_projection = nn.Parameter(
            torch.randn(text_embedding_dim, projection_dim))

    def forward(self, image, input_ids, attention_mask):
        image_features_unnormalized = self.image_encoder(
            image)  # [batch_size, 512]
        text_features_unnormalized = self.text_encoder(
            input_ids, attention_mask)  # [batch_size, 768]

        # [batch_size, 512]
        image_features = image_features_unnormalized @ self.image_projection
        # [batch_size, 512]
        text_features = text_features_unnormalized @ self.text_projection

        # [batch_size, 512]
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        # [batch_size, 512]
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        logits_per_image = image_features @ text_features.T
        logits_per_text = logits_per_image.T

        output = {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'image_features': image_features,
            'text_features': text_features,
            'image_features_unnormalized': image_features_unnormalized,
            'text_features_unnormalized': text_features_unnormalized,
        }
        return output


if __name__ == "__main__":
    clip = CLIPModel(
        text_model_name="distilbert-base-uncased",
        image_model_name="resnet50",
        text_pretrained=True,
        text_trainable=False,
        image_pretrained=True,
        image_trainable=False,
        text_embedding_dim=768,
        image_embedding_dim=2048,
        projection_dim=512,
    )
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 100, (batch_size, 10))
    attention_mask = torch.randint(0, 2, (batch_size, 10))
    logits_per_image, logits_per_text = clip(image, input_ids, attention_mask)
    print(logits_per_image.shape)
    print(logits_per_text.shape)
