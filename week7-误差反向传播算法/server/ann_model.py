import torch
import torch.nn as nn


class AnnModel(nn.Module):
    """
    定义模型（与训练时的模型结构相同）
    """

    def __init__(self):
        super(AnnModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            prob = torch.softmax(x, 1)
            max_index = torch.argmax(prob, 1)
            return torch.argmax(x, 1), prob[0][max_index], prob[0]
