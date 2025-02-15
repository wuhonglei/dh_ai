import torch
import torch.nn as nn
from transformers import BertModel
from torchinfo import summary


class TitleClassifier(nn.Module):
    def __init__(self, num_classes: int, bert_name: str):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # [batch_size, hidden_size]
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


if __name__ == '__main__':
    model = TitleClassifier(num_classes=10, bert_name='bert-base-uncased')
    summary(model, input_data=(torch.randint(
        0, 100, (1, 10)), torch.randint(0, 2, (1, 10))))
