import torch
import torch.nn as nn
from torchinfo import summary


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.hidden = nn.Linear(10, 100)
        self.relu = nn.ReLU()
        self.output = nn.Linear(100, 10)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    model = TestModel()
    # torch.save(model.state_dict(), 'test.pth')
    summary(model, input_size=(100, 10))
