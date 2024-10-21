import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, output_size: int, padding_idx: int):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.RNN(embed_dim, hidden_size,
                          batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        # hidden: [num_layers * num_directions, batch, hidden_size]
        _, hidden = self.rnn(x)
        # concat_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        concat_hidden = hidden.squeeze(0)
        output = self.fc(concat_hidden)
        return output


def get_class_weights(labels: list[str]) -> torch.Tensor:
    # 假设你有一个包含所有训练集标签的 numpy 数组或列表 train_labels
    train_labels = np.array(labels)

    # 获取类别列表
    classes = np.unique(train_labels)

    # 计算类别权重
    class_weights = compute_class_weight(
        class_weight='balanced', classes=classes, y=train_labels)

    # 将 numpy 数组转换为 torch 张量
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights
