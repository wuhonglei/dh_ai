import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import ImageModel
from dataset import ImageDataset
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import wandb
import os

from config import train_csv_path, test_csv_path, label_name, project_name, image_dir


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=world_size,
                            rank=rank)
    return rank, world_size, gpu


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


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


def build_transform():
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def build_loader(csv_path: str, image_dir: str, label_col: str, label_encoder: LabelEncoder, transform: A.Compose, batch_size: int, shuffle: bool, rank: int, world_size: int):
    df = pd.read_csv(csv_path)
    success_df = df[df['download_success'] == 1]
    image_names = success_df['main_image_name'].tolist()
    labels = success_df[label_col].tolist()
    labels = np.array(label_encoder.transform(labels))
    dataset = ImageDataset(image_names, image_dir, labels, transform)

    # Create distributed sampler
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )


def train_one_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, epoch: int, device: torch.device, rank: int, world_size: int):  # type: ignore
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Only show progress bar on rank 0
    progress_bar = tqdm(train_loader, desc=f'训练第 {epoch} 轮', total=len(
        train_loader), disable=rank != 0)

    for i, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

        if rank == 0:
            progress_bar.set_postfix(loss=loss.item())

    # Synchronize metrics across all processes
    total_loss = torch.tensor(running_loss / len(train_loader)).to(device)
    total_acc = torch.tensor(correct / total).to(device)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)

    return total_loss.item() / world_size, total_acc.item() / world_size


def eval_one_epoch(model: nn.Module, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, epoch: int, device: torch.device, rank: int, world_size: int):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f'评估第 {epoch} 轮', total=len(
            test_loader), disable=rank != 0)
        for batch in progress_bar:
            input_ids, labels = batch
            outputs = model(input_ids.to(device))
            loss = criterion(outputs, labels.to(device))
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) ==
                              labels.to(device)).sum().item()
            total_samples += labels.size(0)

    # Synchronize metrics across all processes
    total_loss = torch.tensor(total_loss / len(test_loader)).to(device)
    total_acc = torch.tensor(total_correct / total_samples).to(device)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)

    return total_loss.item() / world_size, total_acc.item() / world_size


def train(_config={}):
    # Initialize distributed training
    rank, world_size, gpu = setup_distributed()
    device = torch.device(f'cuda:{gpu}')

    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(project=project_name, config={**_config})
        config = wandb.config
    else:
        config = _config

    epochs = config['epochs']
    dropout = config['dropout']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    model_name = config['model_name']

    # Adjust batch size for distributed training
    batch_size = batch_size // world_size

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.read_csv(train_csv_path)[label_name])
    transform = build_transform()

    train_loader = build_loader(train_csv_path, image_dir, label_name,
                                label_encoder, transform, batch_size, shuffle=True,
                                rank=rank, world_size=world_size)
    test_loader = build_loader(test_csv_path, image_dir, label_name,
                               label_encoder, transform, batch_size, shuffle=False,
                               rank=rank, world_size=world_size)

    model = ImageModel(num_classes=len(label_encoder.classes_),
                       model_name=model_name, drop_rate=dropout)
    model = model.to(device)
    model = DDP(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(  # type: ignore
        model.parameters(), lr=learning_rate)

    if rank == 0:
        progress_bar = tqdm(range(epochs), desc='训练', total=epochs)
    else:
        progress_bar = range(epochs)

    best_test_acc = 0.0
    for epoch in progress_bar:
        train_loader.sampler.set_epoch(epoch)  # type: ignore
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, device, rank, world_size)
        test_loss, test_acc = eval_one_epoch(
            model, test_loader, criterion, epoch, device, rank, world_size)

        if rank == 0:
            progress_bar.set_postfix(  # type: ignore
                train_loss=train_loss, test_loss=test_loss,
                train_acc=train_acc, test_acc=test_acc)
            best_test_acc = max(best_test_acc, test_acc)
            wandb.log({
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc
            })

    if rank == 0:
        wandb.summary['params_size'] = get_readable_params_size(model.module)
        wandb.summary['best_test_acc'] = best_test_acc

    cleanup_distributed()


def main():
    use_sweep = False
    if not use_sweep:
        config = {
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 12,
            'dropout': 0.5,
            'model_name': 'vgg16'
        }
        train(config)
        return

    sweep_config = {
        'method': 'bayes',
        'name': project_name,
        'parameters': {
            'batch_size': {'values': [64, 128]},
            'learning_rate': {'values': [0.001, 0.0001]},
            'epochs': {'values': [12, 24]},
            'dropout': {'values': [0.5, 0.1]},
            'model_name': {'values': ['resnet101', 'vgg16']},
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, train)


if __name__ == '__main__':
    main()
