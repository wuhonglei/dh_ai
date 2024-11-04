from dataset import TranslateDataset, build_vocab, collate_fn
from seq2seq import Seq2Seq, init_weights, test_translate
from decoder import Decoder
from encoder import Encoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

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
model.apply(init_weights)  # 初始化模型参数
model.load_state_dict(torch.load('./models/seq2seq_0.pth'))

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(
    ignore_index=target_vocab['<pad>'])  # type: ignore

epoch_progress = tqdm(range(num_epochs), leave=True)
for epoch in epoch_progress:
    epoch_progress.set_description(f'epoch: {epoch + 1}')
    model.train()
    batch_progress = tqdm(enumerate(train_loader), leave=False)
    for batch_idx, (b_src, b_target) in batch_progress:
        batch_progress.set_description(
            f'batch: {batch_idx + 1}/{len(train_loader)}')

        b_src = b_src.to(device)
        b_target = b_target.to(device)
        # [batch_size, seq_len, output_size]
        output = model(b_src, b_target, teacher_force_ratio)
        new_output = output[:, 1:].reshape(-1, output.shape[2])
        new_b_target = b_target[:, 1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(new_output, new_b_target)
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        if batch_idx % 10 == 0:
            batch_progress.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), f'./models/seq2seq_{epoch}.pth')
    test_translate(model, src_vocab, target_vocab)
