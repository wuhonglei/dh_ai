import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """
    线性分类器
    """

    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)
