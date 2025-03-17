
class Encoder:
    def __init__(self, vocab_path: str):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = [line.strip() for line in f.readlines()]

    def encode(self, sentence: list[str]):
        return [self.vocab.index(word) if word in self.vocab else self.vocab.index('<unk>') for word in sentence]

    def decode(self, indices: list[int]):
        return [self.vocab[index] for index in indices]

    def __len__(self):
        return len(self.vocab)
