import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq, init_weights

input_size = 100  # 词典大小
output_size = 100  # 词典大小
embed_size = 50  # 词向量维度
hidden_size = 1024  # 隐藏层维度
num_layers = 2  # LSTM层数
dropout = 0.5  # dropout概率
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(input_size, embed_size, hidden_size,
                  num_layers, dropout).to(device)
decoder = Decoder(output_size, embed_size, hidden_size,
                  num_layers, dropout).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)
model.apply(init_weights)  # 初始化模型参数
