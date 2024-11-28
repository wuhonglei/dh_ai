import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

from loss import gram_matrix, compute_content_loss, compute_style_loss


def preprocess_batch(batch):
    batch = batch.clone()
    (b, c, h, w) = batch.size()
    for i in range(b):
        batch[i] = normalize_batch(batch[i])
    return batch


def normalize_batch(batch):
    # 预处理，用于 VGG 网络
    mean = torch.tensor([123.68, 116.779, 103.939]).to(batch.device)
    batch = batch.sub(mean.view(3, 1, 1)).div(255)
    return batch


def train(transformer, vgg, train_loader, style_image, device, epochs=2, style_weight=1e5, content_weight=1e0):
    optimizer = optim.Adam(transformer.parameters(), lr=0.001)  # type: ignore
    vgg.to(device)
    transformer.to(device)

    # 计算风格特征的 Gram 矩阵
    with torch.no_grad():
        style = style_image.to(device)
        style_features = vgg(style)
        style_gram = [gram_matrix(y) for y in style_features]

    for epoch in range(epochs):
        transformer.train()
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)
            y = y.clamp(0, 255)

            # 转换为 VGG 输入
            x_vgg = preprocess_batch(x)
            y_vgg = preprocess_batch(y)

            # 提取特征
            features_x = vgg(x_vgg)
            features_y = vgg(y_vgg)

            # 计算内容损失
            content_loss = content_weight * \
                compute_content_loss(features_y, features_x)

            # 计算风格损失
            style_loss = style_weight * \
                compute_style_loss(features_y, style_gram)

            # 总损失
            total_loss = content_loss + style_loss
            total_loss.backward()

            optimizer.step()

            if (batch_id) % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{batch_id+1}/{len(train_loader)}], Loss: {total_loss.item()}")

        # 每个 epoch 保存一次模型
        torch.save(transformer.state_dict(),
                   f"./models/model_epoch_{epoch+1}.pth")


if __name__ == '__main__':
    from transformer_net import TransformerNet
    from vgg16_net import VGG16

    # 数据集和加载器
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder("./data", transform)
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = TransformerNet().to(device)
    vgg = VGG16().to(device).eval()

    style_image = transform(
        Image.open("style.png").convert("RGB")).unsqueeze(0)  # type: ignore

    train(transformer, vgg, train_loader, style_image, device, epochs=10)
