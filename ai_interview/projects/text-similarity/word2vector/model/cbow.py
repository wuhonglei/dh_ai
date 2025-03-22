from torch import nn
import torch
from torchinfo import summary
from typing import Annotated
from torch import Tensor
from torch.nn import functional as F


class CBOWModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, pad_idx: int):
        super(CBOWModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def encode(self, context_idxs: Annotated[Tensor, "batch_size, context_size"]) -> Annotated[Tensor, "batch_size, embedding_dim"]:
        # [batch_size, context_size, embedding_dim]
        embeds = self.embedding(context_idxs)

        # 创建 mask: padding_idx 位置为 0，其他位置为 1
        mask = (context_idxs != self.embedding.padding_idx).float()
        # 扩展 mask 维度以匹配 embeds
        mask = mask.unsqueeze(-1)

        # 使用 mask 进行加权平均
        context_sum = (embeds * mask).sum(dim=1)  # 只累加非 padding 位置
        context_len = mask.sum(dim=1)  # 计算非 padding 的词数
        context_mean = context_sum / (context_len + 1e-10)  # 添加小值避免除0

        return context_mean

    def forward(self, context_idxs: Annotated[Tensor, "batch_size, context_size"],  target_idx: Annotated[Tensor, "batch_size"] | None = None) -> Annotated[Tensor, "batch_size, vocab_size"]:
        context_mean = self.encode(context_idxs)

        # [batch_size, vocab_size]
        out = self.linear(context_mean)
        if target_idx is not None:
            loss = F.cross_entropy(out, target_idx)
            return loss
        return out


if __name__ == "__main__":
    model = CBOWModel(16439, 100, 0)
    input_data = torch.randint(0, 16439, (1, 100), dtype=torch.long)
    summary(model, input_data=input_data)
