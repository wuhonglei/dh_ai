"""
不使用 RNN 的关键词分类模型
"""

import torch
import torch.nn as nn
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel


class KeywordCategoryModel(nn.Module):
    def __init__(self, bert_model_name: str, hidden_size: int,  num_labels: int, dropout: float = 0.0):
        super(KeywordCategoryModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        config = self.bert.config

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(
            input_ids, attention_mask, token_type_ids=None)
        # 提取[CLS]向量
        # shape: (batch_size, hidden_size)
        pooled_output = bert_outputs.pooler_output

        # features_output = self.fc(cls_output)

        # 将[CLS]位置的向量传入分类头
        cls_output = self.classifier(pooled_output)
        return cls_output
