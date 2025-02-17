import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import StockDataset
from model import StockPredictModel

with open('stock_data.pkl', 'br') as f:
    df = pickle.load(f)

AAPL = df['Adj Close'][ (df['company_name']=='APPLE')]

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total = 0
    total_loss = 0.0
    batch_progress = tqdm(enumerate(dataloader), leave=False)
    for i, (x, y) in batch_progress:
        batch_progress.set_description(f'batch: {i + 1}/{len(dataloader)}')
        total += x.size(0)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_progress.set_postfix(loss=loss.item())
    return total_loss / total

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    for i, (x, y) in enumerate(dataloader):
        total += x.size(0)
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        total_loss += (output - y).pow(2).sum().item()

    rmse = (total_loss / total) ** 0.5
    return rmse



# 划分训练集和测试集
ratio = 0.95
seq_len = 60
training_data_len = int(len(AAPL)*ratio)
train_data = AAPL[:training_data_len]
test_data = AAPL[training_data_len - seq_len:]

train_dataset = StockDataset(train_data, seq_len)
test_dataset = StockDataset(test_data, seq_len)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = StockPredictModel(dropout=0)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epoch_progress = tqdm(range(15))
for epoch in epoch_progress:
    model.train()
    epoch_progress.set_description(f'epoch: {epoch + 1}')
    train(model, train_loader, criterion, optimizer, device)
    rmse = evaluate(model, test_loader, device)
    epoch_progress.set_postfix(rmse=rmse)


torch.save(model.state_dict(), 'stock_predict.pth')