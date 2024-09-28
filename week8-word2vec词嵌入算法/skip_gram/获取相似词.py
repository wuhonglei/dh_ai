from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import pickle

from model import SkipGramModel

checkpoints = torch.load("./models/skip_gram.pth",
                         map_location=torch.device('cpu'), weights_only=True)
vocab_size = checkpoints['vocab_size']
embedding_dim = checkpoints['embedding_dim']
model = SkipGramModel(vocab_size, embedding_dim)
model.load_state_dict(checkpoints['model_state_dict'])

result = model.similar_word(
    '考古', checkpoints['word2idx'], checkpoints['idx2word'])

for word, sim in result:
    print(f"{word}: {sim}")
