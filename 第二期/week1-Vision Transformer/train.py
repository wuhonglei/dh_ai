import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import VisionTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from shutdown import shutdown
import atexit
import wandb

wandb_config = {
    'project': 'Vision Transformer',
    'config': {
        'epochs': 15,  # 训练轮数
        'batch_size': 32,  # 批量大小
        'size': (32, 32),  # 输入图像大小
        'patch_size': 2,  # 分块大小
        'embed_dim': 72,  # 嵌入维度
        'n_heads': 12,  # 注意力头数
        'depth': 12,  # 深度
        'n_classes': 10,  # 类别数
        'lr': 1e-3,  # 学习率
        'drop_rate': 0.,  # 丢弃率
        'attn_drop_rate': 0.,  # 注意力丢弃率
        'qkv_bias': True,  # 偏置
        'mlp_ratio': 4.,  # MLP比例
        'in_channels': 3,  # 输入通道数
        'model_name': 'vit_patch4_32',  # 模型名称
        'shutdown': False,
        'sweep': True,
    },
    'job_type': 'train',
    'tags': ['pretrained:False'],
}

sweep_config = {
    'method': 'grid',  # 调参方法：random / grid / bayes
    'program': 'train.py',
    'metric': {
        'name': 'test_accuracy',  # 优化目标
        'goal': 'maximize'  # 最大化验证集准确率
    },
    'parameters': {
        'n_heads': {
            'values': [8, 12]
        },
        'depth': {
            'values': [8, 12]
        },
    }
}


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total


def train(model, train_loader, test_loader, epochs, device):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # type: ignore

    # 训练模型
    epoch_progress = tqdm(range(epochs), desc="Epoch", leave=True, position=0)
    for epoch in epoch_progress:
        model.train()
        batch_progress = tqdm(train_loader, desc="Batch",
                              leave=False, position=1)
        total_loss = 0
        start_time = time.time()
        train_correct = 0
        train_total = 0
        for images, labels in batch_progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_progress.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        test_correct, test_total = evaluate(model, test_loader, device)
        test_accuracy = test_correct / test_total
        wandb.log({
            'train_loss': avg_loss,
            'test_accuracy': test_accuracy,
            'train_accuracy': train_accuracy,
            'epoch_time': int(time.time() - start_time)
        })
        epoch_progress.set_postfix(
            avg_loss=avg_loss, test_accuracy=f"{100 * test_accuracy:.2f}%", train_accuracy=f"{100 * train_accuracy:.2f}%")

    print("Training complete")


def clean_up(model, config):
    # 保存模型权重
    torch.save(model.state_dict(), f"./models/{config['model_name']}.pth")
    print("Model saved")
    wandb.finish()
    if config['shutdown']:
        shutdown(10)


def main():
    wandb.init(**wandb_config)
    config = wandb.config

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(config['size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 加载 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(
        root='/mnt/dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(
        root='/mnt/dataset', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 定义模型
    model = VisionTransformer(img_size=config['size'][0], patch_size=config['patch_size'], in_channels=config['in_channels'], n_classes=config['n_classes'], embed_dim=config['embed_dim'],
                              depth=config['depth'], n_heads=config['n_heads'], mlp_ratio=config['mlp_ratio'], qkv_bias=config['qkv_bias'], drop_rate=config['drop_rate'], attn_drop_rate=config['attn_drop_rate'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    atexit.register(clean_up, model, config)
    # 训练模型
    train(model, train_loader, test_loader, config['epochs'], device)


if __name__ == "__main__":
    if wandb_config['config']['sweep']:
        sweep_id = wandb.sweep(sweep_config, project='vit-sweep-demo')
        wandb.agent(sweep_id, main)
    else:
        main()
