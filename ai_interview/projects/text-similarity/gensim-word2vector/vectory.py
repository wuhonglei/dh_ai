from vocab import Vocab
from model.cbow import CBOWModel
import torch


class Vector:
    def __init__(self, vocab: Vocab, model: CBOWModel, device: torch.device):
        self.vocab = vocab
        self.model = model
        self.device = device

    def get_embedding(self, sentence: str) -> list[float]:
        words = self.vocab.tokenize(sentence)
        indices = self.vocab.batch_encoder(words)
        tensor_indices = torch.LongTensor(indices).unsqueeze(0).to(self.device)
        content_embedding = self.model.encode(tensor_indices)
        embedding_numpy = content_embedding.detach().cpu().numpy().astype('float32')
        return embedding_numpy.tolist()[0]
