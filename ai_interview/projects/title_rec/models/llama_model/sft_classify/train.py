"""
训练模型
"""

import token
import pandas as pd
from sklearn.calibration import LabelEncoder
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from transformers import AutoTokenizer
from dataset import BaseDataset
from model import BaseModel, build_model
from config import train_csv_path, test_csv_path, columns, label_name, max_length, project_name
from tqdm import tqdm
import wandb
from shutdown import shutdown


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


def build_loader(csv_path: str, column_name: str, label_name: str, batch_size: int, tokenizer, max_length: int, shuffle: bool, label_encoder: LabelEncoder):
    dataset = BaseDataset(csv_path, column_name,
                          label_name, tokenizer, max_length, label_encoder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model: BaseModel, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, epoch: int, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    temp_total = 0
    temp_correct = 0
    progress_bar = tqdm(
        train_loader, desc=f'Epoch {epoch}', total=len(train_loader))
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        temp_total += labels.size(0)
        temp_correct += (logits.argmax(dim=1) ==
                         labels).sum().item()
        if batch_idx % 10 == 0:
            progress_bar.set_postfix(
                batch_loss=loss.item(), batch_acc=temp_correct / temp_total)
    return total_loss / len(train_loader), temp_correct / temp_total


def eval_one_epoch(model: BaseModel, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, device: torch.device, epoch: int) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        progress_bar = tqdm(
            test_loader, desc=f'Epoch {epoch}', total=len(test_loader))
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) ==
                              labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / len(test_loader), total_correct / total_samples


def train(_config: dict = {}):
    wandb.init(project=project_name, config={
        **_config,
        'device': get_device()
    })
    config = wandb.config
    model_name = config['model_name']
    num_classes = config['num_classes']
    batch_size = config['batch_size']
    max_length = config['max_length']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    column_name = config['column_name']
    device = config['device']

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.read_csv(train_csv_path)[label_name])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_loader = build_loader(train_csv_path, column_name, label_name,
                                batch_size=batch_size, tokenizer=tokenizer, max_length=max_length, shuffle=True, label_encoder=label_encoder)
    test_loader = build_loader(test_csv_path, column_name, label_name,
                               batch_size=batch_size, tokenizer=tokenizer, max_length=max_length, shuffle=False, label_encoder=label_encoder)
    model = build_model(num_classes=num_classes,
                        model_name=model_name, pad_token_id=-1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(  # type: ignore
        model.parameters(), lr=learning_rate)
    model.to(device)
    best_acc = 0.0
    progress_bar = tqdm(range(epochs), desc='Epoch')
    for epoch in progress_bar:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, epoch, device)
        eval_loss, eval_acc = eval_one_epoch(
            model, test_loader, criterion, device, epoch)
        best_acc = max(best_acc, eval_acc)
        progress_bar.set_postfix(train_loss=train_loss, eval_loss=eval_loss,
                                 eval_acc=eval_acc)
        wandb.log({
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'eval_acc': eval_acc,
            'train_acc': train_acc
        })
    wandb.summary['best_acc'] = best_acc
    wandb.summary['params_size'] = get_readable_params_size(model)


def main():
    config = {
        'batch_size': 320,
        'learning_rate': 1e-5,
        'epochs': 1,
        'max_length': 28,
        'column_name': 'name',
        'model_name': '../Llama-3.2-1B',
        'num_classes': 30
    }
    train(config)


if __name__ == '__main__':
    main()
