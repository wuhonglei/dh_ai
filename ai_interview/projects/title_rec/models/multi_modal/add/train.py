import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer
from model import MultiModalModel
from dataset import MultiModalDataset
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import wandb
import os
import atexit
from shutdown import shutdown
from config import train_csv_path, test_csv_path, project_name, image_dir


def is_distributed():
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


def setup_distributed(port: int = 29500):
    """Initialize distributed training.

    Args:
        port (int): The port number for distributed training. Default is 29500.
                    Different distributed environments should use different ports.
    """
    if is_distributed():
        rank = int(os.environ['RANK'])  # 当前进程的编号
        world_size = int(os.environ['WORLD_SIZE'])  # 总的进程数
        gpu = int(os.environ['LOCAL_RANK'])  # 当前进程的GPU编号

        # 分布式环境，执行初始化
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://127.0.0.1:{port}',  # 使用本地回环地址和指定端口
            world_size=world_size,
            rank=rank
        )
    else:
        # 单机环境，不执行初始化
        rank = 0
        world_size = 1
        gpu = 0

    return rank, world_size, gpu


def cleanup_distributed():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()


def get_device(gpu: int):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
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


def build_loader(csv_path: str, title_col_name: str, image_col_name: str, label_col_name: str, tokenizer, max_length: int, label_encoder: LabelEncoder, image_dir: str, transform: A.Compose, batch_size: int, shuffle: bool, rank: int, world_size: int):
    dataset = MultiModalDataset(csv_path, title_col_name, image_col_name,
                                label_col_name, tokenizer, max_length, label_encoder, image_dir, transform)

    if is_distributed():
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False if sampler else shuffle
    )


def train_one_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, epoch: int, device: torch.device, rank: int, world_size: int):  # type: ignore
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Only show progress bar on rank 0
    progress_bar = tqdm(train_loader, desc=f'训练第 {epoch} 轮', total=len(
        train_loader), disable=rank != 0)

    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

        if rank == 0:
            progress_bar.set_postfix(loss=loss.item(), acc=correct / total)

    # Synchronize metrics across all processes
    if is_distributed():
        total_loss = torch.tensor(running_loss / len(train_loader)).to(device)
        total_acc = torch.tensor(correct / total).to(device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)

        return total_loss.item() / world_size, total_acc.item() / world_size
    else:
        return running_loss / len(train_loader), correct / total


def eval_one_epoch(model: nn.Module, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, epoch: int, device: torch.device, rank: int, world_size: int):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f'评估第 {epoch} 轮', total=len(
            test_loader), disable=rank != 0)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) ==
                              labels).sum().item()
            total_samples += labels.size(0)

    # Synchronize metrics across all processes
    if is_distributed():
        total_loss = torch.tensor(total_loss / len(test_loader)).to(device)
        total_acc = torch.tensor(total_correct / total_samples).to(device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)

        return total_loss.item() / world_size, total_acc.item() / world_size
    else:
        return total_loss / len(test_loader), total_correct / total_samples


def train(_config={}):
    # Initialize distributed training
    rank, world_size, gpu = setup_distributed()
    device = get_device(gpu)

    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(project=project_name, config={
                   **_config, 'device': device, 'world_size': world_size})
        config = wandb.config
    else:
        config = _config

    epochs = config['epochs']
    dropout = config['drop_rate']
    text_learning_rate = config['text_learning_rate']
    image_learning_rate = config['image_learning_rate']
    classifier_learning_rate = config['classifier_learning_rate']
    batch_size = config['batch_size']
    text_model_name = config['text_model_name']
    image_model_name = config['image_model_name']
    title_col_name = config['title_col_name']
    image_col_name = config['image_col_name']
    label_col_name = config['label_col_name']
    max_length = config['max_length']
    freeze_text_encoder = config['freeze_text_encoder']
    freeze_image_encoder = config['freeze_image_encoder']

    # Adjust batch size for distributed training
    batch_size = batch_size // world_size

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.read_csv(train_csv_path)[label_col_name])
    transform = build_transform()
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    train_loader = build_loader(train_csv_path, title_col_name, image_col_name,
                                label_col_name, tokenizer, max_length, label_encoder, image_dir, transform, batch_size, shuffle=True,
                                rank=rank, world_size=world_size)
    test_loader = build_loader(test_csv_path, title_col_name, image_col_name,
                               label_col_name, tokenizer, max_length, label_encoder, image_dir, transform, batch_size, shuffle=False,
                               rank=rank, world_size=world_size)

    model = MultiModalModel(num_classes=len(label_encoder.classes_),
                            text_model_name=text_model_name,
                            image_model_name=image_model_name,
                            drop_rate=dropout,
                            freeze_text_encoder=freeze_text_encoder,
                            freeze_image_encoder=freeze_image_encoder)
    original_model = model.to(device)
    if is_distributed():
        model = DDP(model, device_ids=[gpu])
    else:
        model = original_model

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': original_model.text_encoder.parameters(), 'lr': text_learning_rate},
        {'params': original_model.image_encoder.parameters(), 'lr': image_learning_rate},
        {'params': original_model.classifier.parameters(), 'lr': classifier_learning_rate}
    ])

    if rank == 0:
        progress_bar = tqdm(range(epochs), desc='训练', total=epochs)
    else:
        progress_bar = range(epochs)

    best_test_acc = 0.0
    for epoch in progress_bar:
        start_time = time.time()
        if is_distributed():
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
                'test_acc': test_acc,
                'time': time.time() - start_time
            })

    if rank == 0:
        wandb.summary['params_size'] = get_readable_params_size(original_model)
        wandb.summary['best_test_acc'] = best_test_acc

    cleanup_distributed()


def cleanup():
    pass
    # shutdown(10)


def main():
    config = {
        'batch_size': 256,
        'text_learning_rate': 2e-5,
        'image_learning_rate': 1e-4,
        'classifier_learning_rate': 1e-4,
        'epochs': 6,
        'drop_rate': 0.5,
        'max_length': 28,
        'text_model_name': 'distilbert-base-uncased',
        'image_model_name': 'resnet101',
        'title_col_name': 'remove_prefix',
        'image_col_name': 'main_image_name',
        'label_col_name': 'level1_global_be_category_id',
        'freeze_text_encoder': False,
        'freeze_image_encoder': False
    }
    train(config)
    return


if __name__ == '__main__':
    atexit.register(cleanup)
    main()
