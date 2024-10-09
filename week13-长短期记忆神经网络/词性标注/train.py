from tqdm import tqdm

import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from torchtext.datasets import UDPOS

from dataset import collate_fn, POSTagDataset
from model import BiLSTMPosTagger, accuracy

with open('text_vocab.pkl', 'rb') as f:
    text_vocab = pickle.load(f)

with open('pos_vocab.pkl', 'rb') as f:
    pos_vocab = pickle.load(f)

train_data, valid_data, test_data = UDPOS(root='.data')
dataset = POSTagDataset(train_data)
dataloader = DataLoader(dataset, batch_size=32,
                        collate_fn=lambda batch: collate_fn(batch, text_vocab, pos_vocab), shuffle=True)

test_dataset = POSTagDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=32,
                             collate_fn=lambda batch: collate_fn(batch, text_vocab, pos_vocab), shuffle=False)

# 超参数
INPUT_DIM = len(text_vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = len(pos_vocab)
N_LAYERS = 2
DROPOUT = 0.25
NUM_EPOCHS = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTMPosTagger(
    INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, bidirectional=True, dropout=DROPOUT
)
model.load_state_dict(torch.load('pos_tagger.pth'))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pos_vocab['<pad>'])


epoch_progress = tqdm(range(NUM_EPOCHS), leave=True)
for epoch in epoch_progress:
    epoch_progress.set_description(f'epoch: {epoch + 1}')

    batch_progress = tqdm(enumerate(dataloader), leave=False)
    for i, (text, pos) in batch_progress:
        model.train()
        batch_progress.set_description(f'batch: {i + 1}/{len(dataloader)}')

        text = text.to(device)
        pos = pos.to(device)

        optimizer.zero_grad()
        """
        text: [batch_size, seq_len]
        pos: [batch_size, seq_len]
        output: [batch_size, seq_len, output_dim]
        """
        output = model(text)
        # output: [batch_size * seq_len, output_dim]
        output = output.view(-1, output.shape[-1])
        # pos: [batch_size * seq_len]
        pos = pos.view(-1)
        loss = criterion(output, pos)
        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        continue

    model.eval()
    total_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for text, pos in test_dataloader:
            text = text.to(device)
            pos = pos.to(device)

            output = model(text)
            output = output.view(-1, output.shape[-1])
            pos = pos.view(-1)
            loss = criterion(output, pos)
            total_loss += loss.item()

            i_correct, i_total = accuracy(output, pos, pos_vocab['<pad>'])
            correct += i_correct
            total += i_total

    epoch_progress.set_postfix(
        val_loss=total_loss/len(dataloader), val_acc=correct/total)

torch.save(model.state_dict(), 'pos_tagger.pth')
