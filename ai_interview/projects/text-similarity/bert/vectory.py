from vocab import Vocab
import torch
from model import EmbeddingModel


class Vector:
    def __init__(self, vocab: Vocab, model: EmbeddingModel, device: torch.device):
        self.vocab = vocab
        self.model = model
        self.device = device

    def get_embedding(self, sentence: str) -> list[float]:
        inputs = self.vocab.tokenize(sentence)
        content_embedding = self.model.encode(
            input_ids=inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].to(self.device))  # type: ignore
        embedding_numpy = content_embedding.detach().cpu().numpy().astype('float32')
        return embedding_numpy.tolist()[0]
