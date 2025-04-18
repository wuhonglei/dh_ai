"""
训练模型
"""

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from transformers import AutoTokenizer
from dataset import BaseDataset
from model import BaseModel
from config import train_csv_path, test_csv_path,  project_name
from tqdm import tqdm
import pandas as pd
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


def build_loader(csv_path: str, column_name: str, label_name: str, batch_size: int, tokenizer, max_length: int, shuffle: bool, label_encoder: LabelEncoder):
    dataset = BaseDataset(csv_path, column_name,
                          label_name, tokenizer, max_length, label_encoder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model: BaseModel, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, epoch: int, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(
        train_loader, desc=f'Epoch {epoch}', total=len(train_loader))
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            progress_bar.set_postfix(batch_loss=loss.item())
    return total_loss / len(train_loader)


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
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) ==
                              labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / len(test_loader), total_correct / total_samples


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
    label_name = config['label_name']

    label_encoder = LabelEncoder()
    label_encoder_df = pd.concat([pd.read_csv(train_csv_path), pd.read_csv(
        test_csv_path)], ignore_index=True)
    label_encoder.fit(label_encoder_df[label_name])
    num_classes = len(label_encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    train_loader = build_loader(train_csv_path, column_name, label_name,
                                batch_size=batch_size, tokenizer=tokenizer, max_length=max_length, shuffle=True, label_encoder=label_encoder)
    test_loader = build_loader(test_csv_path, column_name, label_name,
                               batch_size=batch_size, tokenizer=tokenizer, max_length=max_length, shuffle=False, label_encoder=label_encoder)
    model = BaseModel(num_classes=num_classes, bert_name=bert_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    best_acc = 0.0
    progress_bar = tqdm(range(epochs), desc='Epoch')
    for epoch in progress_bar:
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, epoch, device)
        eval_loss, eval_acc = eval_one_epoch(
            model, test_loader, criterion, device, epoch)
        best_acc = max(best_acc, eval_acc)
        progress_bar.set_postfix(train_loss=train_loss, eval_loss=eval_loss,
                                 eval_acc=eval_acc)
        wandb.log({
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'eval_acc': eval_acc
        })
    wandb.summary['best_acc'] = best_acc
    wandb.summary['params_size'] = get_readable_params_size(model)


def main():
    use_sweep = False
    if not use_sweep:
        config = {
            'batch_size': 128,
            'learning_rate': 3e-5,
            'epochs': 5,
            'max_length': 28,
            'column_name': 'name',
            'bert_name': 'bert-base-uncased',
            'label_name': 'global_be_category_id'
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
            'label_name': {
                'values': ['global_be_category_id']
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, train)


if __name__ == '__main__':
    main()
