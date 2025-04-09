"""
训练模型
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from typing import Callable
from config import train_csv_path, vocab_dir, columns, label_name, test_csv_path
from vocab import Vocab, load_vocab
from dataset import TextCNNDataset
from model import TextCNN
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def get_device():
    return torch.device('cpu')

    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


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
                        total=len(train_loader), position=1, leave=False)
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
                          total=len(test_loader), position=2, leave=False):
            input_ids, labels = batch
            outputs = model(input_ids.to(device))
            loss = criterion(outputs, labels.to(device))
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) ==
                              labels.to(device)).sum().item()
            total_samples += labels.size(0)
    return total_loss / len(test_loader), total_correct / total_samples


def train():
    min_freq = 5
    max_seq_length = 20
    epochs = 3
    column = 'spacy_tokenized_name'
    embedding_dim = 100
    num_filters = 100
    filter_sizes = [3, 4, 5]
    learning_rate = 0.001
    num_classes = 30
    device = get_device()
    batch_size = 1280

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
                        total=epochs, position=0, leave=False)
    for epoch in progress_bar:
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, device)
        test_loss, test_acc = eval_one_epoch(
            model, test_loader, criterion, device)
        progress_bar.set_postfix(
            train_loss=train_loss, test_loss=test_loss, test_acc=test_acc)


if __name__ == '__main__':
    start_time = time.time()
    train()
    end_time = time.time()
    print(f'训练时间: {end_time - start_time:.2f} 秒')
