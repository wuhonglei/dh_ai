from dataset import TranslateDataset, build_vocab, collate_fn
from seq2seq import Seq2Seq, init_weights, test_translate
from decoder import Decoder
from encoder import Encoder
import torch
from torch.utils.data import DataLoader

train_dataset = TranslateDataset('./csv/train.csv')
test_dataset = TranslateDataset('./csv/test.csv')


def collate(batch): return collate_fn(batch, src_vocab, target_vocab)


src_vocab, target_vocab = build_vocab(train_dataset)
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False, collate_fn=collate)


input_size = len(src_vocab)  # 词典大小
output_size = len(target_vocab)  # 词典大小
embed_size = 50  # 词向量维度
hidden_size = 100  # 隐藏层维度
num_layers = 1  # LSTM层数
dropout = 0.5  # dropout概率
teacher_force_ratio = 0.5
num_epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(input_size, embed_size, hidden_size,
                  num_layers, dropout).to(device)
decoder = Decoder(output_size, embed_size, hidden_size,
                  num_layers, dropout).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(torch.load('./models/seq2seq_0.pth'))
