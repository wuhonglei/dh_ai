import torch
import torch.nn as nn
from transformers import AutoModel
from torchinfo import summary


class BaseModel(nn.Module):
    def __init__(self, bert_name: str, num_level1: int, num_leaf: int, dropout: float = 0.1):
        super(BaseModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.num_level1 = num_level1
        self.num_leaf = num_leaf
        self.fc_level1 = nn.Linear(self.bert.config.hidden_size, num_level1)
        self.fc_leaf = nn.Linear(
            self.bert.config.hidden_size + num_level1, num_leaf)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, level1_labels: torch.Tensor | None = None):
        x = self.bert(input_ids, attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        level1_logits = self.fc_level1(x)
        if level1_labels is not None:
            # 训练时使用真实的一级标签
            level1_one_hot = nn.functional.one_hot(
                level1_labels, num_classes=self.num_level1).float()
        else:
            # 推理时使用预测的一级标签
            level1_probs = nn.functional.softmax(level1_logits, dim=-1)
            level1_one_hot = level1_probs

        # 拼接 BERT 输出和一级目录表示
        leaf_input = torch.cat([x, level1_one_hot], dim=-1)
        leaf_logits = self.fc_leaf(leaf_input)

        return level1_logits, leaf_logits


if __name__ == '__main__':
    model = BaseModel(num_level1=10, num_leaf=10,
                      bert_name='distilbert-base-uncased')
    input_data = torch.randint(0, 100, (1, 22))
    attention_mask = torch.ones(1, 22)
    summary(model, input_data=input_data, attention_mask=attention_mask)
