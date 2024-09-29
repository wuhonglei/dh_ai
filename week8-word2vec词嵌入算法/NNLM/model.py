import torch
import torch.nn as nn


class NNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, context_size):
        super(NNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.linear1 = nn.Linear(embed_size * context_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # [batch_size, context_size] -> [batch_size, context_size]
        x = x.view(-1, x.size(-1))
        x = self.embed(x)
        x = x.view(x.size(0), -1)  # [batch_size, context_size * embed_size]
        x = self.activation(self.linear1(x))
        out = self.linear2(x)
        return out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
