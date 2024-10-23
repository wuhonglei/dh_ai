import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCnn(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, kernel_sizes, num_filters, dropout=0.5):
        super(TextCnn, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_size)) for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_length, embed_size)
        x = x.unsqueeze(1)     # (batch_size, 1, seq_length, embed_size)

        # [(batch_size, num_filters, seq_length - k +1)]
        conv_outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_outs = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(
            2) for conv_out in conv_outs]  # [(batch_size, num_filters)]

        # (batch_size, num_filters * len(kernel_sizes))
        out = torch.cat(pooled_outs, 1)
        out = self.dropout(out)
        out = self.fc(out)  # (batch_size, num_classes)
        return out


if __name__ == '__main__':
    model = TextCnn(vocab_size=100, embed_size=100, num_classes=10,
                    kernel_sizes=[2, 3, 4], num_filters=128)
    print(model)
    x = torch.randint(0, 100, (32, 100))
    print(model(x).shape)
