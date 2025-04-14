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
from dataset import TextCNNDataset
from model import TextCNN
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


def build_loader(csv_path: str, column_name: str, label_name: str, batch_size: int, word_to_id: dict[str, int], max_seq_length: int, shuffle: bool):
    dataset = TextCNNDataset(csv_path, column_name,
                             label_name, tokenizer, word_to_id, max_seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model: TextCNN, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, epoch: int, device: torch.device):
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


def eval_one_epoch(model: TextCNN, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, device: torch.device):
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
    num_filters = config['num_filters']
    filter_sizes = config['filter_sizes']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    num_classes = config['num_classes']

    vocab = Vocab()
    word_to_id, _ = vocab.load_vocab_freq(
        os.path.join(vocab_dir, f'{column}.csv'), min_freq)
    train_loader = build_loader(
        train_csv_path, column, label_name, batch_size, word_to_id, max_seq_length, shuffle=True)
    test_loader = build_loader(
        test_csv_path, column, label_name, batch_size, word_to_id, max_seq_length, shuffle=False)

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        filter_sizes=filter_sizes,
        num_classes=num_classes,
        padding_idx=vocab.padding_idx
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
    use_sweep = False
    if not use_sweep:
        config = {
            'batch_size': 640,
            'learning_rate': 0.001,
            'epochs': 8,
            'embedding_dim': 300,
            'num_filters': 100,
            'filter_sizes': [3, 4, 5],
            'min_freq': 3,
            'max_seq_length': 25,
            'column': 'remove_prefix',
            'num_classes': 30
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
                'values': [5, 10]
            },
            'embedding_dim': {
                'values': [100, 200, 300]
            },
            'num_filters': {
                'values': [100, 150]
            },
            'filter_sizes': {
                'values': [[3, 4, 5]]
            },
            'min_freq': {
                'values': [3, 5]
            },
            'max_seq_length': {
                'values': [15, 20]
            },
            'column': {
                'values': ['spacy_tokenized_name', 'remove_spacy_stop_words', 'remove_prefix', 'remove_prefix_emoji', 'remove_prefix_emoji_symbol', 'remove_prefix_emoji_symbol_stop_words']
            },
            'num_classes': {
                'values': [30]
            }
        }
    }

    if os.environ.get('sweep_id'):
        sweep_id: str = os.environ.get('sweep_id', '')
    else:
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        write_local_env('sweep_id', sweep_id)
    wandb.agent(sweep_id, train)


if __name__ == '__main__':
    main()
