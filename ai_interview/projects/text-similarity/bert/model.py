from transformers import BertModel, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, model_name: str, max_position_embeddings: int, use_projection: bool):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            model_name,
            max_position_embeddings=max_position_embeddings
        )
        self.use_projection = use_projection
        if use_projection:
            hidden_size = self.bert.config.hidden_size
            self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.bert(input_ids, attention_mask)
        # 获取 [CLS] 位置的输出 [batch_size, hidden_dim]
        cls_output = output.last_hidden_state[:, 0, :]
        if self.use_projection:
            cls_output = self.projection(cls_output)
        return cls_output

    def forward_pair(self, input_ids_1: torch.Tensor, attention_mask_1: torch.Tensor, input_ids_2: torch.Tensor, attention_mask_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 获取两个句子的 [CLS] 位置的输出, 形状 (batch_size, hidden_size)
        output_1 = self.forward(input_ids_1, attention_mask_1)
        output_2 = self.forward(input_ids_2, attention_mask_2)
        return output_1, output_2


def compute_loss(output_1: torch.Tensor, output_2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    # 计算相似度矩阵
    logits = output_1 @ output_2.T / temperature

    # 创建标签（对角线位置为正样本）
    labels = torch.arange(logits.shape[0], device=logits.device)

    return F.cross_entropy(logits, labels)

    # # 计算对称的loss
    # loss = (F.cross_entropy(logits, labels) +
    #         F.cross_entropy(logits.T, labels)) / 2

    # return loss
