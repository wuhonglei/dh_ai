"""
训练模型
"""

import pandas as pd
import json
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from transformers import AutoTokenizer
from dataset import BaseDataset
from model import BaseModel
from config import train_csv_path, test_csv_path,  project_name, leaf_level_map_path
from tqdm import tqdm
import wandb
from shutdown import shutdown
from sklearn.calibration import LabelEncoder


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_readable_params_size(model: nn.Module):
    """
    估算模型保存后的体积(MB)
    假设所有参数都是 float32 类型（4字节）
    """
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    # 计算存储大小（字节）
    size_bytes = num_params * 4  # float32 每个参数占用 4 字节
    # 转换为 MB
    size_mb = size_bytes / 1024 / 1024
    return size_mb


def transform_level_map(json_path: str, level1_label_encoder: LabelEncoder, leaf_label_encoder: LabelEncoder) -> dict[int, list[int]]:
    with open(json_path, 'r') as f:
        level_map = json.load(f)

    level_map_dict = {}
    for level1_label, leaf_labels in level_map.items():
        level1_label_id = level1_label_encoder.transform(
            [level1_label]).item()  # type: ignore
        level_map_dict[level1_label_id] = []
        for leaf_label in leaf_labels:
            leaf_label_id = leaf_label_encoder.transform(
                [leaf_label]).item()  # type: ignore
            level_map_dict[level1_label_id].append(leaf_label_id)
    return level_map_dict


def build_loader(csv_path: str, column_name: str, level1_label_name: str, leaf_label_name: str, batch_size: int, tokenizer, max_length: int, shuffle: bool, level1_label_encoder: LabelEncoder, leaf_label_encoder: LabelEncoder):
    dataset = BaseDataset(csv_path, column_name, level1_label_name,
                          leaf_label_name, tokenizer, max_length, level1_label_encoder, leaf_label_encoder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model: BaseModel, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, epoch: int, device: torch.device) -> tuple[float, float]:
    model.train()
    level1_total_loss = 0.0
    leaf_total_loss = 0.0
    progress_bar = tqdm(
        train_loader, desc=f'Epoch {epoch}', total=len(train_loader))
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        level1_labels = batch['level1'].to(device)
        leaf_labels = batch['leaf'].to(device)
        level1_logits, leaf_logits = model(
            input_ids, attention_mask, level1_labels)
        loss_level1 = criterion(level1_logits, level1_labels)
        loss_leaf = criterion(leaf_logits, leaf_labels)
        loss = loss_level1 + loss_leaf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        level1_total_loss += loss_level1.item()
        leaf_total_loss += loss_leaf.item()
        if batch_idx % 10 == 0:
            progress_bar.set_postfix(batch_loss=loss.item(), level1_loss=loss_level1.item(),
                                     leaf_loss=loss_leaf.item())

    return level1_total_loss / len(train_loader), leaf_total_loss / len(train_loader)


def eval_one_epoch(model: BaseModel, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, device: torch.device, epoch: int, level_map: dict[int, list[int]]) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    level1_correct = 0
    leaf_correct = 0
    total_samples = 0
    with torch.no_grad():
        progress_bar = tqdm(
            test_loader, desc=f'Epoch {epoch}', total=len(test_loader))
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            level1_labels = batch['level1'].to(device)
            leaf_labels = batch['leaf'].to(device)
            level1_logits, leaf_logits = model(
                input_ids, attention_mask)
            loss_level1 = criterion(level1_logits, level1_labels)
            loss_leaf = criterion(leaf_logits, leaf_labels)
            loss = loss_level1 + loss_leaf
            total_loss += loss.item()

            level1_pred = level1_logits.argmax(dim=1)
            level1_correct += (level1_pred == level1_labels).sum().item()

            if not level_map:
                leaf_correct += (leaf_logits.argmax(dim=1)
                                 == leaf_labels).sum().item()
                total_samples += level1_labels.size(0)
                continue

            # 初始化叶子节点预测的累积结果
            all_leaf_preds = []

            # 遍历每个样本
            for level1_pred_idx, leaf_logit in zip(level1_pred, leaf_logits):
                # 获取 level1_pred 对应的叶子节点的索引
                leaf_pred_indices = level_map[level1_pred_idx.item()]

                # 获取叶子节点的索引对应的 logits
                leaf_pred_logits = leaf_logit[leaf_pred_indices]
                # 获取叶子节点的索引对应的预测（局部索引）
                local_leaf_pred = leaf_pred_logits.argmax(dim=0).item()
                # 将局部索引转换为全局索引
                global_leaf_pred = leaf_pred_indices[local_leaf_pred]

                all_leaf_preds.append(global_leaf_pred)

            # 计算叶子节点的正确预测数
            leaf_correct += sum(pred == label for pred,
                                label in zip(all_leaf_preds, leaf_labels)).item()

            total_samples += level1_labels.size(0)
    return total_loss / len(test_loader), level1_correct / total_samples, leaf_correct / total_samples


def train(_config: dict = {}):
    wandb.init(project=project_name, config={
        **_config,
        'device': get_device()
    })
    config = wandb.config
    bert_name = config['bert_name']
    batch_size = config['batch_size']
    max_length = config['max_length']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    column_name = config['column_name']
    device = config['device']
    level1_label_name = config['level1_label_name']
    leaf_label_name = config['leaf_label_name']
    dropout = config['dropout']

    level1_label_encoder = LabelEncoder()
    label_encoder_df = pd.concat([pd.read_csv(train_csv_path), pd.read_csv(
        test_csv_path)], ignore_index=True)
    level1_label_encoder.fit(label_encoder_df[level1_label_name])
    leaf_label_encoder = LabelEncoder()
    leaf_label_encoder.fit(label_encoder_df[leaf_label_name])

    num_level1 = len(level1_label_encoder.classes_)
    num_leaf = len(leaf_label_encoder.classes_)
    level_map = transform_level_map(
        leaf_level_map_path, level1_label_encoder, leaf_label_encoder)

    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    train_loader = build_loader(train_csv_path, column_name, level1_label_name,
                                leaf_label_name, batch_size=batch_size, tokenizer=tokenizer, max_length=max_length, shuffle=True, level1_label_encoder=level1_label_encoder, leaf_label_encoder=leaf_label_encoder)
    test_loader = build_loader(test_csv_path, column_name, level1_label_name,
                               leaf_label_name, batch_size=batch_size, tokenizer=tokenizer, max_length=max_length, shuffle=False, level1_label_encoder=level1_label_encoder, leaf_label_encoder=leaf_label_encoder)
    model = BaseModel(num_level1=num_level1,
                      num_leaf=num_leaf, bert_name=bert_name, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(  # type: ignore
        model.parameters(), lr=learning_rate)
    model.to(device)
    best_level1_acc = 0.0
    best_leaf_acc = 0.0
    progress_bar = tqdm(range(epochs), desc='Epoch')
    for epoch in progress_bar:
        train_level1_loss, train_leaf_loss = train_one_epoch(model, train_loader, criterion,
                                                             optimizer, epoch, device)
        eval_loss, level1_acc, leaf_acc = eval_one_epoch(
            model, test_loader, criterion, device, epoch, {})
        best_level1_acc = max(best_level1_acc, level1_acc)
        best_leaf_acc = max(best_leaf_acc, leaf_acc)
        progress_bar.set_postfix(train_level1_loss=train_level1_loss, train_leaf_loss=train_leaf_loss, eval_loss=eval_loss,
                                 level1_acc=level1_acc, leaf_acc=leaf_acc)
        wandb.log({
            'train_level1_loss': train_level1_loss,
            'train_leaf_loss': train_leaf_loss,
            'eval_loss': eval_loss,
            'level1_acc': level1_acc,
            'leaf_acc': leaf_acc
        })
    wandb.summary['best_level1_acc'] = best_level1_acc
    wandb.summary['best_leaf_acc'] = best_leaf_acc
    wandb.summary['params_size'] = get_readable_params_size(model)


def main():
    use_sweep = False
    if not use_sweep:
        config = {
            'batch_size': 128,
            'learning_rate': 3e-5,
            'epochs': 6,
            'max_length': 50,
            'column_name': 'name',
            'bert_name': 'bert-base-uncased',
            'level1_label_name': 'level1_global_be_category_id',
            'leaf_label_name': 'global_be_category_id',
            'dropout': 0.1
        }
        train(config)
        return

    sweep_config = {
        'method': 'bayes',
        'name': 'base_model',
        'count': 100,  # 限制最多运行100次
        'metric': {
            'name': 'eval_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'batch_size': {
                'values': [128, 256]
            },
            'learning_rate': {
                'values': [3e-5, 5e-5]
            },
            'epochs': {
                'values': [3]
            },
            'max_length': {
                'values': [18, 22, 28]
            },
            'column_name': {
                'values': ['name', 'spacy_tokenized_name', 'remove_spacy_stop_words', 'remove_prefix', 'remove_prefix_emoji', 'remove_prefix_emoji_symbol', 'remove_prefix_emoji_symbol_stop_words']
            },
            'bert_name': {
                'values': ['bert-base-uncased', 'distilbert-base-uncased', 'albert-base-v1', 'albert-xlarge-v1', 'albert-xlarge-v2']
            },
            'level1_label_name': {
                'values': ['level1_global_be_category_id']
            },
            'leaf_label_name': {
                'values': ['global_be_category_id']
            },
            'dropout': {
                'values': [0.1, 0.2, 0.3]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, train)


if __name__ == '__main__':
    main()
