from numpy import dtype
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from torchinfo import summary


class SiameseNetwork(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, projection_dim: int, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_size=hidden_dim,
                            batch_first=True,
                            bidirectional=False)
        self.projection = nn.Linear(hidden_dim, projection_dim)

    def load_pretrained_embedding_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")

        state_dict = torch.load(path, map_location='cpu')
        self.embedding.weight.data.copy_(state_dict['embedding.weight'])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(input_ids)
        # 输出形状 [batch_size, seq_len, hidden_dim]
        output, (hidden, _) = self.lstm(embedded)
        new_hidden = hidden.squeeze(0)
        return self.projection(new_hidden)

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


if __name__ == "__main__":
    model = SiameseNetwork(vocab_size=10000, embedding_dim=200,
                           hidden_dim=256, projection_dim=256, pad_idx=0)
    input_ids = torch.randint(0, 10000, (2, 16), dtype=torch.long)
    summary(model, input_data=input_ids)
