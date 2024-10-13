import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import StockDataset
from model import StockPredictModel

with open('stock_data.pkl', 'br') as f:
    df = pickle.load(f)

data = df[df['company_name']=='APPLE']
AAPL = data['Adj Close']


# 划分训练集和测试集
ratio = 0.95
seq_len = 60
training_data_len = int(len(AAPL)*ratio)
train_data = AAPL[:training_data_len]
test_data = AAPL[training_data_len - seq_len:]

train_dataset = StockDataset(train_data, seq_len)
test_dataset = StockDataset(test_data, seq_len)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = StockPredictModel()
model.to(device)
model.load_state_dict(torch.load('stock_predict.pth'))
predictions = []
model.eval()
for i, (x, y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
      output = model(x)
    predictions.extend(output)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = np.array(predictions)
# Visualize the data
plt.figure(figsize=(16, 6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()