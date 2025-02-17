import torch
import random
import numpy as np
from torch.utils.data import Dataset


class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=5, negative_samples=5):
        self.corpus = corpus
        self.vocab = vocab
        self.window_size = window_size
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sentence = self.corpus[idx]
        words = sentence.strip().split()
        word_indices = [self.vocab.word2idx[word]
                        for word in words if word in self.vocab.word2idx]
        pairs = []
        for i, center in enumerate(word_indices):
            window = random.randint(1, self.window_size)
            context_indices = word_indices[max(
                0, i - window): i] + word_indices[i + 1: i + window + 1]
            pairs.append((center, context_indices))
        return pairs

    def collate_fn(self, batch):
        centers = []
        contexts = []
        negatives = []
        for pairs in batch:
            for center, context_indices in pairs:
                for context in context_indices:
                    centers.append(center)
                    contexts.append(context)
                    neg_samples = []
                    while len(neg_samples) < self.negative_samples:
                        neg_sample = np.random.choice(
                            self.vocab.word_probs)
                        if neg_sample not in context_indices:
                            neg_samples.append(neg_sample)
                    negatives.append(neg_samples)
        return torch.LongTensor(centers), torch.LongTensor(contexts), torch.LongTensor(negatives)
