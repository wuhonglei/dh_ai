from vocab import Vocab
from model import SiameseNetwork
import torch


class Vector:
    def __init__(self, vocab: Vocab, model: SiameseNetwork, device: torch.device):
        self.vocab = vocab
        self.model = model
        self.device = device

    def get_embedding(self, sentence: str, max_length: int) -> list[float]:
        words = self.vocab.tokenize(sentence)
        indices = self.vocab.batch_encoder(words)
        indices = indices[:max_length] + \
            [self.vocab.pad_idx] * (max_length - len(indices))
        tensor_indices = torch.LongTensor(indices).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor_indices)
        embedding_numpy = output.detach().cpu().numpy().astype('float32')
        return embedding_numpy.tolist()[0]
