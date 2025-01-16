import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import atexit
import wandb
import time
from torchinfo import summary

wandb_config = {
    'project': 'Vision Transformer',
    'config': {
        'epochs': 10,
        'size': (224, 224),
        'batch_size': 32,
        'lr': 1e-3,
        'drop_rate': 0.,
        'attn_drop_rate': 0.,
        'qkv_bias': True,
        'mlp_ratio': 4.,
        'depth': 12,
        'n_heads': 12,
        'embed_dim': 768,
        'patch_size': 16,
        'in_channels': 3,
        'n_classes': 10,
        'model_name': 'vit_224_16_pretrained',
        'shutdown': False,
    },
    'job_type': 'train',
    'tags': ['pretrained:True'],
}

wandb.init(**wandb_config)
config = wandb.config

# 加载预训练的 ViT 模型
# 设置模型下载路径
os.environ['TORCH_HOME'] = '/mnt/model/huggingface/hub/'
model = timm.create_model('vit_base_patch16_224',  # type: ignore
                          pretrained=True)

model.reset_classifier(config['n_classes'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 加载数据集
transform = transforms.Compose([
    transforms.Resize(config['size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(root='/mnt/dataset', train=True,
                                 download=True, transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True)

test_dataset = datasets.CIFAR10(root='/mnt/dataset', train=False,
                                download=True, transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=False)


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total


def train(model, train_loader, epochs):
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])  # type: ignore

    # 训练模型
    epoch_progress = tqdm(range(epochs), desc="Epoch")
    for epoch in epoch_progress:
        model.train()
        batch_progress = tqdm(train_loader, desc="Batch")
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
        test_correct, test_total = evaluate(model, test_loader)
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


def clean_up():
    # 保存模型权重
    torch.save(model.state_dict(), f"{config['model_name']}.pth")
    print("Model saved")
    wandb.finish()


if __name__ == "__main__":
    atexit.register(clean_up)
    train(model, train_loader, epochs=config['epochs'])
    # summary(model, input_size=(1, 3, 224, 224))
