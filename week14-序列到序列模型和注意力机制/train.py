import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encoder import Encoder
from decoder import Decoder
from seq2seq import Seq2Seq, init_weights

from dataset import TranslateDataset, build_vocab, collate_fn

train_dataset = TranslateDataset('./csv/train.csv')
valid_dataset = TranslateDataset('./csv/validation.csv')
test_dataset = TranslateDataset('./csv/test.csv')


def collate(batch): return collate_fn(batch, src_vocab, target_vocab)


src_vocab, target_vocab = build_vocab(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=collate)
valid_loader = DataLoader(valid_dataset, batch_size=32,
                          shuffle=False, collate_fn=collate)  # 验证集和测试集不需要shuffle
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False, collate_fn=collate)

for batch_idx, (src, target) in enumerate(train_loader):
    print(src.shape, target.shape)
    break


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

# 创建 batch_size=3, seq_len=7 的输入
src = torch.randint(0, input_size, (3, 7)).to(device)

# 创建 batch_size=3, seq_len=5 的输出
trg = torch.randint(0, output_size, (3, 5)).to(device)

teacher_force_ratio = 0.5
outputs = model(src, trg, teacher_force_ratio)

print(outputs.shape)  # torch.Size([3, 5, 100])
