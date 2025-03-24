from transformers import BertModel, AutoTokenizer
import torch


class EmbeddingModel(torch.nn.Module):
    def __init__(self, model_name: str, max_position_embeddings: int):
        super().__init__()
        self.model_name = model_name
        self.max_position_embeddings = max_position_embeddings
        self.model = BertModel.from_pretrained(
            model_name,
            max_position_embeddings=max_position_embeddings
        )  # type: ignore

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids, attention_mask)
        # 获取 [CLS] 位置的输出
        cls_output = output.last_hidden_state[:, 0, :]
        return cls_output
