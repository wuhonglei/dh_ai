import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from torchinfo import summary


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, trainable: bool):
        super().__init__()
        if pretrained:
            self.bert = DistilBertModel.from_pretrained(model_name)
            if not trainable:
                for param in self.bert.parameters():
                    param.requires_grad = False
        else:
            config = DistilBertConfig()
            self.bert = DistilBertModel(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output


if __name__ == "__main__":
    text_encoder = TextEncoder(
        model_name="distilbert-base-uncased", pretrained=True, trainable=False)
    print(text_encoder)

    input_ids = torch.randint(0, 100, (1, 10))
    attention_mask = torch.randint(0, 2, (1, 10))
    input_data = (input_ids, attention_mask)
    summary(text_encoder, input_data=input_data)
