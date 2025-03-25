from vocab import Vocab
from model import CBOWModel


class Vector:
    def __init__(self, vocab: Vocab, model: CBOWModel):
        self.vocab = vocab
        self.model = model

    def get_embedding(self, sentence: str) -> list[float]:
        words = self.vocab.tokenize(sentence)
        content_embedding = self.model.encode(words)
        return content_embedding.tolist()
