import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import Union

class StockDataset(Dataset):
    def __init__(self, stock_prices: list[Union[float, int]], seq_len: int = 60):
        self.seq_len = seq_len
        self.data = []
        self.label = []
        for i in range(seq_len, len(stock_prices)):
            self.data.append(stock_prices[i - seq_len:i])
            self.label.append(stock_prices[i])
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]
    

if __name__ == '__main__':
    stock_prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dataset = StockDataset(stock_prices, 3)  # type: ignore
    print(dataset[0]) # (tensor([1., 2., 3.]), tensor(4.))
    print(dataset[1]) # (tensor([2., 3., 4.]), tensor(5.))
    print(dataset[2]) # (tensor([3., 4., 5.]), tensor(6.))
    print(dataset[3]) # (tensor([4., 5., 6.]), tensor(7.))
    print(dataset[4]) # (tensor([5., 6., 7.]), tensor(8.))
    print(dataset[5]) # (tensor([6., 7., 8.]), tensor(9.))
    print(dataset[6]) # (tensor([7., 8., 9.]), tensor(10.))
    print(len(dataset)) # 7