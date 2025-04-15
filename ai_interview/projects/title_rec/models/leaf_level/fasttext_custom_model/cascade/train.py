"""
训练模型
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from config import train_csv_path, vocab_dir, test_csv_path, project_name
from vocab import Vocab
from dataset import FastTextDataset
from model import FastText
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from dotenv import load_dotenv
import pandas as pd
from sklearn.preprocessing import LabelEncoder

local_env_path = '.env.local'
load_dotenv(local_env_path)


def write_local_env(key: str, value: str):
    with open(local_env_path, 'a') as f:
        f.write(f'{key}={value}\n')


def get_device():
    if torch.cuda.is_available():
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


def tokenizer(text: str) -> list[str]:
    return text.split()


def build_loader(csv_path: str, column_name: str, level1_label_name: str, leaf_label_name: str, batch_size: int, word_to_id: dict[str, int], max_seq_length: int, shuffle: bool, wordNgrams: int, level1_label_encoder: LabelEncoder, leaf_label_encoder: LabelEncoder):
    dataset = FastTextDataset(csv_path, column_name,
                              level1_label_name, leaf_label_name, tokenizer, word_to_id, max_seq_length, wordNgrams, level1_label_encoder, leaf_label_encoder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model: FastText, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, epoch: int, device: torch.device):
    model.train()
    level1_total_loss = 0.0
    leaf_total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'训练第 {epoch} 轮',
                        total=len(train_loader))
    for batch in progress_bar:
        input_ids, level1_labels, leaf_labels = batch
        outputs = model(input_ids.to(device), level1_labels.to(device))
        loss_level1 = criterion(outputs[0], level1_labels.to(device))
        loss_leaf = criterion(outputs[1], leaf_labels.to(device))
        loss = loss_level1 + loss_leaf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        level1_total_loss += loss_level1.item()
        leaf_total_loss += loss_leaf.item()
        progress_bar.set_postfix(loss_level1=loss_level1.item(),
                                 loss_leaf=loss_leaf.item())
    return level1_total_loss / len(train_loader), leaf_total_loss / len(train_loader)


def eval_one_epoch(model: FastText, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, device: torch.device):
    model.eval()
    total_loss = 0.0
    level1_correct = 0
    leaf_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='测试',
                          total=len(test_loader)):
            input_ids, level1_labels, leaf_labels = batch
            level1_logits, leaf_logits = model(input_ids.to(device))
            loss_level1 = criterion(level1_logits, level1_labels.to(device))
            loss_leaf = criterion(leaf_logits, leaf_labels.to(device))
            total_loss += loss_level1.item() + loss_leaf.item()
            level1_pred = level1_logits.argmax(dim=1)
            leaf_pred = leaf_logits.argmax(dim=1)
            level1_correct += (level1_pred == level1_labels).sum().item()
            leaf_correct += (leaf_pred == leaf_labels).sum().item()
            total_samples += leaf_labels.size(0)
    return total_loss / len(test_loader), level1_correct / total_samples, leaf_correct / total_samples


def train(_config: dict = {}):
    # 默认配置
    device = get_device()

    # 初始化 wandb
    wandb.init(project=project_name, config={
        **_config,
    })
    # 使用 wandb.config 更新配置，并提供默认值
    config = wandb.config
    min_freq = config['min_freq']
    max_seq_length = config['max_seq_length']
    epochs = config['epochs']
    column = config['column']
    embedding_dim = config['embedding_dim']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    wordNgrams = config['wordNgrams']
    dropout = config['dropout']
    use_relu = config['use_relu']
    level1_label_name = config['level1_label_name']
    leaf_label_name = config['leaf_label_name']

    level1_label_encoder = LabelEncoder()
    label_encoder_df = pd.concat([pd.read_csv(train_csv_path), pd.read_csv(
        test_csv_path)], ignore_index=True)
    level1_label_encoder.fit(label_encoder_df[level1_label_name])
    leaf_label_encoder = LabelEncoder()
    leaf_label_encoder.fit(label_encoder_df[leaf_label_name])

    num_level1 = len(level1_label_encoder.classes_)
    num_leaf = len(leaf_label_encoder.classes_)

    vocab = Vocab()
    word_to_id, _ = vocab.load_vocab_freq(
        os.path.join(vocab_dir, f'{column}.csv'), min_freq)
    train_loader = build_loader(
        train_csv_path, column, level1_label_name, leaf_label_name, batch_size, word_to_id, max_seq_length, shuffle=True, wordNgrams=wordNgrams, level1_label_encoder=level1_label_encoder, leaf_label_encoder=leaf_label_encoder)
    test_loader = build_loader(
        test_csv_path, column, level1_label_name, leaf_label_name, batch_size, word_to_id, max_seq_length, shuffle=False, wordNgrams=wordNgrams, level1_label_encoder=level1_label_encoder, leaf_label_encoder=leaf_label_encoder)

    model = FastText(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        padding_idx=vocab.padding_idx,
        num_level1=num_level1,
        num_leaf=num_leaf,
        dropout=dropout,
        use_relu=use_relu
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(epochs), desc='训练',
                        total=epochs)
    best_level1_acc = 0.0
    best_leaf_acc = 0.0
    for epoch in progress_bar:
        train_level1_loss, train_leaf_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, device)
        eval_loss, level1_acc, leaf_acc = eval_one_epoch(
            model, test_loader, criterion, device)
        progress_bar.set_postfix(
            train_level1_loss=train_level1_loss, train_leaf_loss=train_leaf_loss, eval_loss=eval_loss, level1_acc=level1_acc, leaf_acc=leaf_acc)
        best_level1_acc = max(best_level1_acc, level1_acc)
        best_leaf_acc = max(best_leaf_acc, leaf_acc)
        wandb.log({
            'train_level1_loss': train_level1_loss,
            'train_leaf_loss': train_leaf_loss,
            'eval_loss': eval_loss,
            'level1_acc': level1_acc,
            'leaf_acc': leaf_acc
        })

    wandb.summary['params_size'] = get_readable_params_size(model)
    wandb.summary['best_level1_acc'] = best_level1_acc
    wandb.summary['best_leaf_acc'] = best_leaf_acc


def main():
    use_sweep = False
    if not use_sweep:
        config = {
            'batch_size': 128,
            'learning_rate': 0.001,
            'epochs': 12,
            'embedding_dim': 150,
            'min_freq': 3,
            'max_seq_length': 40,
            'column': 'remove_spacy_stop_words',
            'wordNgrams': 2,
            'dropout': 0,
            'use_relu': False,
            'level1_label_name': 'level1_global_be_category_id',
            'leaf_label_name': 'global_be_category_id',
        }
        train(config)
        return

    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'test_loss',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 3,
            'eta': 2
        },
        'parameters': {
            'batch_size': {
                'values': [128, 256]
            },
            'learning_rate': {
                'values': [0.001]
            },
            'epochs': {
                'values': [10, 15]
            },
            'embedding_dim': {
                'values': [100, 200]
            },
            'min_freq': {
                'values': [2, 3]
            },
            'max_seq_length': {
                'values': [30, 40]
            },
            'column': {
                'values': ['spacy_tokenized_name', 'remove_spacy_stop_words']
            },
            'wordNgrams': {
                'values': [2]
            },
            'dropout': {
                'values': [0, 0.2]
            },
            'use_relu': {
                'values': [False]
            },
            'level1_label_name': {
                'values': ['level1_global_be_category_id']
            },
            'leaf_label_name': {
                'values': ['global_be_category_id']
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    write_local_env('sweep_id', sweep_id)
    wandb.agent(sweep_id, train)


if __name__ == '__main__':
    main()
