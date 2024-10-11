import torch
import torch.nn as nn

from torchinfo import summary


class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.LSTM(
            embed_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        """
        input: [batch_size]
        hidden: [num_layers, batch_size, hidden_size]
        cell: [num_layers, batch_size, hidden_size]
        embedding: [1, batch_size, embed_size]
        output: [1, batch_size, hidden_size]
        """
        input = input.unsqueeze(0)
        embedding = self.dropout(self.embedding(input))

        """
        output: [1, batch_size, hidden_size]
        hidden: [num_layers, batch_size, hidden_size]
        cell: [num_layers, batch_size, hidden_size]
        """
        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        """
        prediction: [1, batch_size, output_size]
        """
        prediction = self.fc(output.squeeze(0))

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
