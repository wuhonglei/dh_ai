import torch
import torch.nn as nn
import numpy as np
import json
import os
from sklearn.utils.class_weight import compute_class_weight
from .common import make_dir


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


def save_training_json(params: dict[str, int], path: str):
    with open(path, "w") as f:
        f.write(json.dumps(params, indent=4))


def load_training_json(path: str) -> dict[str, int]:
    with open(path, "r") as f:
        return json.loads(f.read())


class EarlyStopping:
    def __init__(self, enable=True, patience=5, min_delta=0.001):
        self.enable = enable
        self.patience = patience
        self.delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if not self.enable:
            return False

        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss > self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def save_model(model_path: str, model: nn.Module):
    make_dir(model_path)
    torch.save(model.state_dict(), model_path)


def scaled_dot_product_attention(Q, K, V, mask=None):
    '''
    计算缩放点积注意力
    Q, K, V: [batch_size, num_heads, seq_len, d_k]
    mask: [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
    '''
    d_k = Q.size(-1)
    # [batch_size, num_heads, seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, d_k]
    return output, attn


def create_pad_mask(seq, pad_idx):
    """
    序列掩码（Padding Mask）
    用于在注意力计算中屏蔽填充（padding）位置
    1 表示未填充位置，0 表示填充位置
    """
    # [batch_size, 1, 1, seq_len]
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_subsequent_mask(size):
    """
    未来位置掩码（Subsequent Mask）
    用于在解码器中屏蔽未来的位置，防止信息泄露。
    1 表示当前及之前位置，0 表示未来位置
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0
