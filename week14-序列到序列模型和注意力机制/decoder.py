import torch
import torch.nn as nn

from torchinfo import summary
from attention import Attention


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.LSTM(
            embed_size + hidden_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.attention = Attention(hidden_size)

    def forward(self, input, hidden, cell, encoder_output):
        """
        input: [batch_size]
        hidden: [num_layers, batch_size, hidden_size]
        cell: [num_layers, batch_size, hidden_size]
        embedding: [1, batch_size, embed_size]
        output: [1, batch_size, hidden_size]
        encoder_output: [seq_len, batch_size, hidden_size]
        """
        input = input.unsqueeze(0)
        embedding = self.dropout(self.embedding(input))

        attention_weights = self.attention(hidden[-1], encoder_output)
        context = attention_weights.bmm(encoder_output.transpose(0, 1))
        context = context.transpose(0, 1)
        lstm_input = torch.cat((embedding, context), dim=2)

        """
        output: [1, batch_size, hidden_size]
        hidden: [num_layers, batch_size, hidden_size]
        cell: [num_layers, batch_size, hidden_size]
        """
        output, (hidden, cell) = self.rnn(lstm_input, (hidden, cell))

        """
        prediction: [1, batch_size, output_size]
        """
        combined = torch.cat((output.squeeze(0), context.squeeze(0)), dim=1)
        prediction = self.fc(combined)

        return prediction, hidden, cell


if __name__ == '__main__':
    output_size = 100  # 词典大小
    embed_size = 50  # 词向量维度
    hidden_size = 1024  # 隐藏层维度
    num_layers = 2  # LSTM层数
    p = 0.5  # dropout概率

    decoder = Decoder(output_size, embed_size, hidden_size, num_layers, p)
    print(decoder)
    input = torch.randint(0, 1, (1,))
    hidden = torch.zeros(num_layers, 1, hidden_size)
    cell = torch.zeros(num_layers, 1, hidden_size)
    prediction, hidden, cell = decoder(input, hidden, cell)
    # summary(decoder, input_size=(1,), dtypes=[torch.long], device='cpu', col_names=[
    #         "input_size", "output_size", "num_params"])
    print(prediction.shape, hidden.shape, cell.shape)

    for i in range(5):
        prediction, hidden, cell = decoder(input, hidden, cell)
        print(prediction.shape, hidden.shape, cell.shape)
        input = prediction.argmax(1)
        print(input)
        print(prediction.argmax(1))
        print(prediction.argmax(1).item())
        print(prediction.argmax(1).item() == 0)
        if prediction.argmax(1).item() == 0:
            break
