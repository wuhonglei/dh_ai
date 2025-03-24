from transformers import BertModel, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, model_name: str, max_position_embeddings: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            model_name,
            max_position_embeddings=max_position_embeddings
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.bert(input_ids, attention_mask)
        # 获取 [CLS] 位置的输出
        cls_output = output.last_hidden_state[:, 0, :]
        return cls_output

    def forward(self, input_ids_1: torch.Tensor, attention_mask_1: torch.Tensor, input_ids_2: torch.Tensor, attention_mask_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 获取两个句子的 [CLS] 位置的输出, 形状 (batch_size, hidden_size)
        output_1 = self.encode(input_ids_1, attention_mask_1)
        output_2 = self.encode(input_ids_2, attention_mask_2)

        logits = output_1 @ output_2.T
        labels = torch.arange(logits.shape[0]).to(logits.device)
        loss = F.cross_entropy(logits, labels)

        return logits, loss
