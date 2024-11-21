import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CNNModel
from dataset import CaptchaDataset
from typing import Dict


def decode_predictions(preds, padding_index: int):
    # preds 的形状为 (seq_len, batch_size, num_classes)
    preds = preds.permute(1, 0, 2)  # 转为 (batch_size, seq_len, num_classes)
    preds = torch.argmax(preds, dim=-1)  # 在类别维度取最大值，得到索引
    preds = preds.cpu().numpy()
    decoded_results: list[list[int]] = []
    for pred in preds:
        # 去除连续重复的索引和空白符（索引 0）
        chars: list[int] = []
        prev_idx = None
        for idx in pred:
            idx = int(idx)
            if idx != prev_idx and idx != padding_index:
                chars.append(idx)
            prev_idx = idx
        decoded_results.append(chars)
    return decoded_results


def evaluate(data_dir: str, model_path: str, captcha_length: int, class_num: int, padding_index, width: int, height: int, characters: str):
    model = CNNModel(width, height, captcha_length, class_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True))
    model.to(device)
    return evaluate_model(data_dir, model, captcha_length, class_num, padding_index, width, height, characters)


def evaluate_model(data_dir, model, captcha_length, class_num, padding_index, width, height, characters):
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    loader_config = {'num_workers': 3, 'pin_memory': True} if is_cuda else {}
    eval_dataset = CaptchaDataset(
        data_dir, captcha_length=captcha_length, characters=characters, padding_index=padding_index, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=2,
                             collate_fn=lambda x: x, **loader_config)

    loss_sum = 0.0
    correct = 0
    total = 0
    criterion = nn.CTCLoss(blank=padding_index)

    model.eval()
    for batch in eval_loader:
        # 数据预处理
        images, labels, label_lengths = zip(*batch)
        # (batch_size, channels, height, width)
        b_images = torch.stack(images).to(device)
        batch_size = b_images.size(0)

        # 将标签拼接成一维向量
        b_labels = torch.cat(labels).to(device)

        # 输入序列长度，假设所有序列长度相同
        input_lengths = torch.full(
            size=(batch_size,), fill_value=(b_images.size(-1) // 4 - 1), dtype=torch.long)
        label_lengths = torch.tensor(
            label_lengths, dtype=torch.long).to(device)

        # 前向传播
        preds = model(b_images)

        # 计算 CTC 损失
        loss = criterion(preds, b_labels, input_lengths, label_lengths)
        loss_sum += loss.item()

        decoded_results = decode_predictions(preds, padding_index)
        for decoded, label in zip(decoded_results, labels):
            total += 1
            if decoded == label.tolist():
                correct += 1

    test_loss = loss_sum / len(eval_loader)
    test_accuracy = 1.0 * correct / (total)

    return test_loss, test_accuracy


if __name__ == '__main__':
    test_loss, test_accuracy = evaluate(data_dir='./data/train-3363-stable-new/test', model_path='./models/model.pth',
                                        characters='0123456789abcdefghijklmnopqrstuvwxyz',
                                        captcha_length=6, class_num=37, padding_index='36', width=96, height=32)

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
