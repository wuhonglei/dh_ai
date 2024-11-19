import torch
import torch.nn as nn
from Transformer import Transformer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from dataset import TranslateDataset, build_vocab, collate_fn
from evaluate import translate_sentence
from utils.model import save_model
from shutdown import shutdown

train_dataset = TranslateDataset('./csv/train.csv', use_cache=True)
valid_dataset = TranslateDataset('./csv/validation.csv', use_cache=True)
test_dataset = TranslateDataset('./csv/test.csv', use_cache=True)


def collate(batch): return collate_fn(batch, src_vocab, target_vocab)


batch_size = 32
src_vocab, target_vocab = build_vocab(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=collate)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                          shuffle=False, collate_fn=collate)  # 验证集和测试集不需要shuffle
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, collate_fn=collate)


# 假设源语言和目标语言的词汇表大小
src_vocab_size = len(src_vocab)  # 词典大小
tgt_vocab_size = len(target_vocab)  # 词典大小
pad_idx: int = src_vocab['<pad>']  # 假设填充符的索引为 0 # type: ignore

# 定义模型参数
d_model = 512
num_layers = 6
num_heads = 8
d_ff = 2048
dropout = 0.1
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型
model = Transformer(src_vocab_size, tgt_vocab_size,
                    d_model, num_layers, num_heads, d_ff, dropout)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
epoch_progress = tqdm(range(num_epochs), leave=True)
for epoch in epoch_progress:
    epoch_progress.set_description(f'epoch: {epoch + 1}')
    model.train()
    batch_progress = tqdm(enumerate(train_loader), leave=False)
    for batch_idx, (b_src, b_target, src_mask, tgt_mask) in batch_progress:
        batch_progress.set_description(
            f'batch: {batch_idx + 1}/{len(train_loader)}')

        b_src = b_src.to(device)
        b_target = b_target.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        # 前向传播
        output = model(b_src, b_target[:, :-1],
                       src_mask, tgt_mask[:, :, :-1, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        b_target = b_target[:, 1:].contiguous().view(-1)

        # 计算损失
        loss = criterion(output, b_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_progress.set_postfix(loss=loss.item())

    src_sentence = 'i like math .'
    translated = translate_sentence(
        model, src_sentence, src_vocab, target_vocab, device)
    print(f'src: {src_sentence}, tgt: {translated}')
    if num_epochs > 1:
        save_model(model, f'./models/transformer_{epoch + 1}.pth')

save_model(model, './models/transformer.pth')
shutdown(10)
