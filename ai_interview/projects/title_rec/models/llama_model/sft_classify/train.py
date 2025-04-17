"""
训练模型
"""
from shutdown import shutdown
import wandb
from tqdm import tqdm
from config import train_csv_path, test_csv_path, columns, label_name, max_length, project_name
from model import BaseModel, build_model
from dataset import BaseDataset
from transformers import AutoTokenizer
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from sklearn.calibration import LabelEncoder
import os
import token
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    if world_size > 1:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo',
                                init_method='env://',
                                world_size=world_size,
                                rank=rank)

    return rank, world_size, device


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


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


def build_loader(csv_path: str, column_name: str, label_name: str, batch_size: int, tokenizer, max_length: int, shuffle: bool, label_encoder: LabelEncoder, rank: int, world_size: int):
    dataset = BaseDataset(csv_path, column_name,
                          label_name, tokenizer, max_length, label_encoder)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)


def train_one_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, epoch: int, device: torch.device, rank: int, world_size: int) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    temp_total = 0
    temp_correct = 0
    train_loader.sampler.set_epoch(epoch)

    progress_bar = tqdm(
        train_loader, desc=f'Epoch {epoch}', total=len(train_loader), disable=rank != 0)
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
        temp_correct += (logits.argmax(dim=1) == labels).sum().item()

        if rank == 0 and batch_idx % 10 == 0:
            progress_bar.set_postfix(
                batch_loss=loss.item(), batch_acc=temp_correct / temp_total)

    # Synchronize metrics across all processes
    if dist.is_initialized():
        dist.all_reduce(torch.tensor(total_loss).to(device))
        dist.all_reduce(torch.tensor(temp_total).to(device))
        dist.all_reduce(torch.tensor(temp_correct).to(device))
        total_loss = total_loss / world_size
        temp_total = temp_total / world_size
        temp_correct = temp_correct / world_size

    return total_loss / len(train_loader), temp_correct / temp_total


def eval_one_epoch(model: nn.Module, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, device: torch.device, epoch: int, rank: int, world_size: int) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(
            test_loader, desc=f'Epoch {epoch}', total=len(test_loader), disable=rank != 0)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    # Synchronize metrics across all processes
    if dist.is_initialized():
        dist.all_reduce(torch.tensor(total_loss).to(device))
        dist.all_reduce(torch.tensor(total_correct).to(device))
        dist.all_reduce(torch.tensor(total_samples).to(device))
        total_loss = total_loss / world_size
        total_correct = total_correct / world_size
        total_samples = total_samples / world_size

    return total_loss / len(test_loader), total_correct / total_samples


def train(_config: dict = {}):
    rank, world_size, device = setup_distributed()

    if rank == 0:
        wandb.init(project=project_name, config={
            **_config,
            'device': device,
            'world_size': world_size
        })
        config = wandb.config
    else:
        config = _config

    model_name = config['model_name']
    num_classes = config['num_classes']
    batch_size = config['batch_size']
    max_length = config['max_length']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    column_name = config['column_name']

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.read_csv(train_csv_path)[label_name])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = build_loader(train_csv_path, column_name, label_name,
                                batch_size=batch_size, tokenizer=tokenizer, max_length=max_length,
                                shuffle=True, label_encoder=label_encoder, rank=rank, world_size=world_size)
    test_loader = build_loader(test_csv_path, column_name, label_name,
                               batch_size=batch_size, tokenizer=tokenizer, max_length=max_length,
                               shuffle=False, label_encoder=label_encoder, rank=rank, world_size=world_size)

    model = build_model(num_classes=num_classes,
                        model_name=model_name, pad_token_id=tokenizer.pad_token_id)
    model = model.to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[device])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_acc = 0.0
    progress_bar = tqdm(range(epochs), desc='Epoch', disable=rank != 0)
    for epoch in progress_bar:
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, epoch, device, rank, world_size)
        eval_loss, eval_acc = eval_one_epoch(
            model, test_loader, criterion, device, epoch, rank, world_size)

        if rank == 0:
            best_acc = max(best_acc, eval_acc)
            progress_bar.set_postfix(train_loss=train_loss, eval_loss=eval_loss,
                                     eval_acc=eval_acc)
            wandb.log({
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc,
                'train_acc': train_acc
            })

    if rank == 0:
        wandb.summary['best_acc'] = best_acc
        wandb.summary['params_size'] = get_readable_params_size(
            model.module if isinstance(model, DDP) else model)

    cleanup_distributed()


def main():
    config = {
        'batch_size': 32,
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
