"""
训练模型
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from typing import Callable
from config import train_csv_path, vocab_dir, columns, label_name, test_csv_path, project_name
from vocab import Vocab, load_vocab
from dataset import FastTextDataset
from model import FastText
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

local_env_path = '.env.local'
load_dotenv(local_env_path)


def write_local_env(key: str, value: str):
    with open(local_env_path, 'a') as f:
        f.write(f'{key}={value}\n')


def get_device():
    return torch.device('cpu')

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


def tokenizer(text: str) -> list[str]:
    return text.split()


def build_loader(csv_path: str, column_name: str, label_name: str, batch_size: int, word_to_id: dict[str, int], max_seq_length: int, shuffle: bool, wordNgrams: int):
    dataset = FastTextDataset(csv_path, column_name,
                              label_name, tokenizer, word_to_id, max_seq_length, wordNgrams)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model: FastText, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, epoch: int, device: torch.device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'训练第 {epoch} 轮',
                        total=len(train_loader))
    for batch in progress_bar:
        input_ids, labels = batch
        outputs = model(input_ids.to(device))
        loss = criterion(outputs, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(train_loader)


def eval_one_epoch(model: FastText, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, device: torch.device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='测试',
                          total=len(test_loader)):
            input_ids, labels = batch
            outputs = model(input_ids.to(device))
            loss = criterion(outputs, labels.to(device))
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) ==
                              labels.to(device)).sum().item()
            total_samples += labels.size(0)
    return total_loss / len(test_loader), total_correct / total_samples


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
    num_classes = config['num_classes']
    wordNgrams = config['wordNgrams']
    dropout = config['dropout']
    use_relu = config['use_relu']

    vocab = Vocab()
    word_to_id, _ = vocab.load_vocab_freq(
        os.path.join(vocab_dir, f'{column}.csv'), min_freq)
    train_loader = build_loader(
        train_csv_path, column, label_name, batch_size, word_to_id, max_seq_length, shuffle=True, wordNgrams=wordNgrams)
    test_loader = build_loader(
        test_csv_path, column, label_name, batch_size, word_to_id, max_seq_length, shuffle=False, wordNgrams=wordNgrams)

    model = FastText(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        padding_idx=vocab.padding_idx,
        num_classes=num_classes,
        dropout=dropout,
        use_relu=use_relu
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(epochs), desc='训练',
                        total=epochs)
    best_test_acc = 0.0
    for epoch in progress_bar:
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, device)
        test_loss, test_acc = eval_one_epoch(
            model, test_loader, criterion, device)
        progress_bar.set_postfix(
            train_loss=train_loss, test_loss=test_loss, test_acc=test_acc)
        best_test_acc = max(best_test_acc, test_acc)
        wandb.log({
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

    wandb.summary['params_size'] = get_readable_params_size(model)
    wandb.summary['best_test_acc'] = best_test_acc


def main():
    use_sweep = True
    if not use_sweep:
        config = {
            'batch_size': 128,
            'learning_rate': 0.001,
            'epochs': 100,
            'embedding_dim': 100,
            'min_freq': 2,
            'max_seq_length': 50,
            'column': 'remove_spacy_stop_words',
            'num_classes': 30,
            'wordNgrams': 2,
            'dropout': 0.5,
            'use_relu': True
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
                'values': [640, 1280]
            },
            'learning_rate': {
                'values': [0.001]
            },
            'epochs': {
                'values': [5, 10, 15]
            },
            'embedding_dim': {
                'values': [100, 200, 300]
            },
            'min_freq': {
                'values': [2, 3, 5]
            },
            'max_seq_length': {
                'values': [30, 40, 50]
            },
            'column': {
                'values': ['spacy_tokenized_name', 'remove_spacy_stop_words']
            },
            'num_classes': {
                'values': [30]
            },
            'wordNgrams': {
                'values': [2]
            },
            'dropout': {
                'values': [0, 0.1, 0.5]
            },
            'use_relu': {
                'values': [True, False]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    write_local_env('sweep_id', sweep_id)
    wandb.agent(sweep_id, train)


if __name__ == '__main__':
    main()
