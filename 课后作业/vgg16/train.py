import torch
import torch.nn as nn


def train(model, train_loader, test_loader, epochs, device):
    """ 训练模型 """
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    for epoch in range(epochs):
        model.train()
        correct = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            correct += (output.argmax(1) == target).sum().item()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        print(
            f"Epoch: {epoch + 1}, Loss: {loss.item()}, Accuracy: {correct / len(train_loader.dataset)}")

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    print("Model has been saved.")


if __name__ == '__main__':
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # 加载数据
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将灰度图像转换为 3 通道
    ])
    train_dataset = datasets.MNIST(
        root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(
        root='data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 加载模型
    from model import VGG16

    model = VGG16()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 训练模型
    train(model, train_loader, test_loader, epochs=1, device=device)
