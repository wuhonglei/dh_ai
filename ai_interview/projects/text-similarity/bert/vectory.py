from vocab import Vocab
import torch
from model import SiameseNetwork
from typing import Literal


class Vector:
    def __init__(self, type: Literal["title", "content"], vocab: Vocab, model: SiameseNetwork, device: torch.device):
        self.type = type
        self.vocab = vocab
        self.model = model
        self.device = device

    def get_embedding(self, sentence: str) -> list[float]:
        inputs = self.vocab.tokenize(sentence)
        forward = self.model.forward_title if self.type == "title" else self.model.forward_content
        content_embedding = forward(
            input_ids=inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].to(self.device))
        embedding_numpy = content_embedding.detach().cpu().numpy().astype('float32')
        return embedding_numpy.tolist()[0]

    def get_embeddings(self, sentences: list[str]) -> list[list[float]]:
        inputs = self.vocab.batch_encoder(sentences)
        forward = self.model.forward_title if self.type == "title" else self.model.forward_content
        content_embeddings = forward(
            input_ids=inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].to(self.device))
        embedding_numpy = content_embeddings.detach().cpu().numpy().astype('float32')
        return embedding_numpy.tolist()
