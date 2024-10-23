import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight


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
