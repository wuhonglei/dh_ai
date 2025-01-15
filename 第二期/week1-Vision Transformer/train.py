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
import time

wandb.init(**{'project': 'Vision Transformer', 'config': {'epochs': 5}})
config = wandb.config

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 加载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(
    root='/mnt/dataset', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(
    root='/mnt/dataset', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义模型
model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, n_classes=10, embed_dim=768,
                          depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def test(model, test_loader):
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    epoch_progress = tqdm(range(epochs), desc="Epoch")
    for epoch in epoch_progress:
        model.train()
        batch_progress = tqdm(train_loader, desc="Batch")
        total_loss = 0
        start_time = time.time()
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
        avg_loss = total_loss / len(train_loader)
        correct, total = test(model, test_loader)
        test_accuracy = correct / total
        wandb.log({
            'train_loss': avg_loss,
            'test_accuracy': test_accuracy,
            'epoch_time': int(time.time() - start_time)
        })
        epoch_progress.set_postfix(
            avg_loss=avg_loss, accuracy=f"{100 * test_accuracy:.2f}%")

    print("Training complete")


def clean_up():
    # 保存模型权重
    torch.save(model.state_dict(), f"model.pth")
    print("Model saved")
    wandb.finish()
    shutdown(10)


if __name__ == "__main__":
    atexit.register(clean_up)
    train(model, train_loader, epochs=config['epochs'])
