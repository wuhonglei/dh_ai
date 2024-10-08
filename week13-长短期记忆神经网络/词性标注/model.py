import torch
import torch.nn as nn


class BiLSTMPosTagger(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, output_dim: int, n_layers: int, bidirectional: bool, dropout: float):
        super().__init__()
        dropout = 0 if n_layers == 1 else dropout
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        """
        text = [batch size, sent len]
        embedded = [batch size, sent len, emb dim]
        """
        embedded = self.dropout(self.embedding(text))
        output, (_, _) = self.lstm(embedded)
        output = self.fc(self.dropout(output))
        return output


if __name__ == '__main__':
    INPUT_DIM = 8866  # 有8866个文本词汇
    EMBEDDING_DIM = 100  # 词向量维度是100
    HIDDEN_DIM = 128  # 隐藏层神经元个数
    OUTPUT_DIM = 19  # 输出层维度，表示有19个标注词
    N_LAYERS = 2  # 隐藏层数量
    DROPOUT = 0.25  # 丢弃比率
    model = BiLSTMPosTagger(
        INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, bidirectional=True, dropout=DROPOUT
    )
    print(model)
