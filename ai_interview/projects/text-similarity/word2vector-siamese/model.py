import torch.nn as nn
import torch
import torch.nn.functional as F
import os


class SiameseNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, projection_dim: int, pad_idx: int, idf_dict: dict[int, float]):
        super(SiameseNetwork, self).__init__()
        self.pad_idx = pad_idx
        self.idf_dict = idf_dict
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.projection = nn.Linear(embedding_dim, projection_dim)

    def load_pretrained_embedding_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")

        state_dict = torch.load(path, map_location='cpu')
        self.embedding.weight.data.copy_(state_dict['embedding.weight'])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_ids)
        # 使用 idf 加权
        idf_weights = torch.tensor(
            [self.idf_dict[idx.item()] for idx in input_ids.flatten()], device=input_ids.device)  # type: ignore
        idf_weights = idf_weights.view_as(input_ids)
        idf_weights_norm = F.normalize(idf_weights, p=2, dim=-1)
        idf_embedded = embedded * idf_weights_norm.unsqueeze(-1)

        idf_embedded_sum = idf_embedded.sum(dim=1)

        # [batch_size, embedding_dim] -> [batch_size, projection_dim]
        output = self.projection(idf_embedded_sum)
        return output

    def forward_pair(self, input_ids_1: torch.Tensor, input_ids_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output_1 = self.forward(input_ids_1)
        output_2 = self.forward(input_ids_2)
        return output_1, output_2


def compute_loss(output_1: torch.Tensor, output_2: torch.Tensor, temperature: float) -> torch.Tensor:
    # 计算相似度矩阵
    logits = output_1 @ output_2.T / temperature

    # 创建标签（对角线位置为正样本）
    labels = torch.arange(logits.shape[0], device=logits.device)

    # 计算对称的loss
    loss = (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2

    return loss
