from transformers import BertModel, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, model_name: str, max_position_embeddings: int, projection_dim: int):
        super().__init__()
        self.bert_title = BertModel.from_pretrained(
            model_name,
            max_position_embeddings=max_position_embeddings
        )
        self.bert_content = BertModel.from_pretrained(
            model_name,
            max_position_embeddings=max_position_embeddings
        )
        self.hidden_size = self.bert_title.config.hidden_size
        self.projection_dim = projection_dim
        self.projection_title = nn.Linear(
            self.hidden_size, self.projection_dim)
        self.projection_content = nn.Linear(
            self.hidden_size, self.projection_dim)

    def forward_text(self, bert: BertModel, projection: nn.Linear, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = bert(input_ids, attention_mask)
        # 获取 [CLS] 位置的输出 [batch_size, hidden_dim]
        cls_output = output.last_hidden_state[:, 0, :]
        cls_output = projection(cls_output)
        return cls_output

    def forward_title(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward_text(self.bert_title, self.projection_title, input_ids, attention_mask)

    def forward_content(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward_text(self.bert_content, self.projection_content, input_ids, attention_mask)

    def forward_pair(self, input_ids_title: torch.Tensor, attention_mask_title: torch.Tensor, input_ids_content: torch.Tensor, attention_mask_content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 获取两个句子的 [CLS] 位置的输出, 形状 (batch_size, hidden_size)
        output_title = self.forward_title(
            input_ids_title, attention_mask_title)
        output_content = self.forward_content(
            input_ids_content, attention_mask_content)
        return output_title, output_content


def compute_loss(output_title: torch.Tensor, output_content: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    # 计算相似度矩阵
    logits = output_title @ output_content.T / temperature

    # 创建标签（对角线位置为正样本）
    labels = torch.arange(logits.shape[0], device=logits.device)

    # 计算对称的loss
    loss = (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2

    return loss
