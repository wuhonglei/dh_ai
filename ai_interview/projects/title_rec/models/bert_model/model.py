import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel, AutoModel
from torchinfo import summary


class BaseModel(nn.Module):
    def __init__(self, bert_name: str, num_classes: int, dropout: float = 0.1):
        super(BaseModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = BaseModel(num_classes=10, bert_name='bert-base-uncased')
    input_data = torch.randint(0, 100, (1, 22))
    attention_mask = torch.ones(1, 22)
    summary(model, input_data=input_data, attention_mask=attention_mask)
