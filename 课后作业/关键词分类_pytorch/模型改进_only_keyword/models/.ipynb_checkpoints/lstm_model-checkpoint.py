import torch
import torch.nn as nn
from torchinfo import summary


class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, output_size: int, padding_idx: int):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout1 = nn.Dropout(0.15)
        self.lstm = nn.LSTM(embed_dim, hidden_size,
                            batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)
        self.dropout2 = nn.Dropout(0.35)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout1(x)
        """
        hidden: [num_layers * num_directions, batch, hidden_size]

        对于 2 layers 和 bidirectional LSTM，hidden 包含以下内容：
        hidden[0]：第一层正向的最后隐藏状态。
        hidden[1]：第一层反向的最后隐藏状态。
        hidden[2]：第二层正向的最后隐藏状态。
        hidden[3]：第二层反向的最后隐藏状态。

        output: [batch, seq_len, hidden_size * num_directions]
        正向层：output[:, -1, :hidden_size] 确实是正向层的最后一个时间步的输出。
        反向层：output[:, 0, hidden_size:] 才是反向层的最后一个时间步的输出，因为反向层是从序列末尾往前处理的。
        """
        _, (hidden, _) = self.lstm(x)
        hidden = self.dropout2(hidden)
        # 第二层正向, 反向的最后隐藏状态 [batch, hidden_size * 2]
        last_layer_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        concat_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        output = self.fc1(concat_hidden)
        output = self.fc2(output)
        return output

# （可选）重新初始化其他层的权重


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


def init_model(DEVICE, model):
    state_dict = torch.load(
        './models/weights/SG_LSTM_128*2_fc_1_model.pth', map_location=DEVICE)
    model.embedding.weight.data.copy_(state_dict['embedding.weight'])
    model.lstm.apply(init_weights)
    model.fc1.apply(init_weights)
    model.fc2.apply(init_weights)


if __name__ == "__main__":
    vocab_size = 50
    embed_dim = 25
    hidden_size = 128
    output_size = 26
    padding_idx = 0
    model = KeywordCategoryModel(
        vocab_size, embed_dim, hidden_size, output_size, padding_idx)
    print(model)
    # summary(model, input_size=(1, 10))
