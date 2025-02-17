import torch
import torch.nn as nn
from torchinfo import summary

class StockPredictModel(nn.Module):
    def __init__(self, dropout=0.2):
        super(StockPredictModel, self).__init__()
        self.lstm = nn.LSTM(1, 128, batch_first=True, num_layers=2) 
        self.fc1 = nn.Linear(128, 25)
        self.fc2 = nn.Linear(25, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        _, (hidden,_) = self.lstm(x)
        hidden = self.dropout(hidden[-1])
        x = self.fc1(hidden)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.squeeze(-1)
        return x
    

if __name__ == '__main__':
    model = StockPredictModel()
    print(model)
    summary(model, input_size=(1, 60), col_names=[
        "input_size", "output_size", "num_params",
    ])