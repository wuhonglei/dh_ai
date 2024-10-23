"""
不使用 RNN 的关键词分类模型
"""

import torch
import torch.nn as nn
from transformers import BertModel


class KeywordCategoryModel(nn.Module):
    def __init__(self, bert_model_name: str, hidden_size: int,  num_labels: int, dropout: float = 0.0):
        super(KeywordCategoryModel, self).__init__()
        self.bert = BertModel.from_pretrained(
            bert_model_name)

        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 分类头
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids, attention_mask)
        # 提取[CLS]向量
        # shape: (batch_size, hidden_size)
        cls_output = bert_outputs.pooler_output

        features_output = self.fc(cls_output)

        # 将[CLS]位置的向量传入分类头
        cls_output = self.classifier(features_output)
        return cls_output
