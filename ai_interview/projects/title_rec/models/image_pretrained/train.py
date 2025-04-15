import torch
import torch.nn as nn
import torch.optim as optim
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

from config import train_csv_path, test_csv_path, label_name, project_name, image_dir


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


def build_loader(csv_path: str, image_dir: str, label_col: str, label_encoder: LabelEncoder, transform: A.Compose, batch_size: int, shuffle: bool):
    df = pd.read_csv(csv_path)
    success_df = df[df['download_success'] == 1]
    image_names = success_df['main_image_name'].tolist()
    labels = success_df[label_col].tolist()
    labels = np.array(label_encoder.transform(labels))
    dataset = ImageDataset(image_names, image_dir, labels, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model: ImageModel, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, epoch: int, device: torch.device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc=f'训练第 {epoch} 轮',
                        total=len(train_loader))
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
        progress_bar.set_postfix(loss=loss.item())
    return running_loss / len(train_loader), correct / total


def eval_one_epoch(model: ImageModel, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, epoch: int, device: torch.device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'评估第 {epoch} 轮',
                          total=len(test_loader)):
            input_ids, labels = batch
            outputs = model(input_ids.to(device))
            loss = criterion(outputs, labels.to(device))
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) ==
                              labels.to(device)).sum().item()
            total_samples += labels.size(0)
    return total_loss / len(test_loader), total_correct / total_samples


def train(_config={}):
    # 默认配置
    device = get_device()
    # 初始化 wandb
    wandb.init(project=project_name, config={
        **_config,
    })
    # 使用 wandb.config 更新配置，并提供默认值
    config = wandb.config
    epochs = config['epochs']
    dropout = config['dropout']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    model_name = config['model_name']

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.read_csv(train_csv_path)[label_name])
    transform = build_transform()
    train_loader = build_loader(train_csv_path, image_dir, label_name,
                                label_encoder, transform, batch_size, shuffle=True)
    test_loader = build_loader(test_csv_path, image_dir, label_name,
                               label_encoder, transform, batch_size, shuffle=False)

    model = ImageModel(num_classes=len(
        label_encoder.classes_), model_name=model_name, drop_rate=dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(epochs), desc='训练',
                        total=epochs)
    best_test_acc = 0.0
    for epoch in progress_bar:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, device)
        test_loss, test_acc = eval_one_epoch(
            model, test_loader, criterion, epoch, device)
        progress_bar.set_postfix(
            train_loss=train_loss, test_loss=test_loss, train_acc=train_acc, test_acc=test_acc)
        best_test_acc = max(best_test_acc, test_acc)
        wandb.log({
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        })
    wandb.summary['params_size'] = get_readable_params_size(model)
    wandb.summary['best_test_acc'] = best_test_acc


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
