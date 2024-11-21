import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from model import CNNModel
from dataset import CaptchaDataset
from utils import get_wandb_config, EarlyStopping, load_config
from evaluate import evaluate_model


def train(train_dir: str, test_dir: str, batch_size: int, pretrained: bool, epochs: int, learning_rate: float, captcha_length: int, class_num: int, characters: str, padding_index: int, model_path: str, width: int, height: int, log: bool, early_stopping={}):
    if log:
        wandb.init(**get_wandb_config(), job_type='train')

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((height, width)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    loader_config = {'num_workers': 3, 'pin_memory': True} if is_cuda else {}
    train_dataset = CaptchaDataset(
        train_dir, captcha_length=captcha_length, characters=characters, padding_index=padding_index, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x, **loader_config)

    model = CNNModel(width, height, captcha_length, class_num)
    if pretrained and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 定义 CTC 损失函数
    # 我们在 characters 中将 '-' 作为 blank，对应索引 0
    criterion = nn.CTCLoss(blank=padding_index)

    optimizer = optim.Adam(  # type: ignore
        model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(**early_stopping)

    epoch_progress = tqdm(range(epochs), desc='Epoch')
    for epoch in epoch_progress:
        loss_sum = 0.0
        acc_sum = 0.0
        start_time = time.time()
        model.train()
        batch_progress = tqdm(enumerate(train_loader), total=len(
            train_loader), desc='Batch', leave=False)
        for batch_ids, batch in batch_progress:
            # 数据预处理
            images, labels, label_lengths = zip(*batch)
            # (batch_size, channels, height, width)
            images = torch.stack(images).to(device)
            batch_size = images.size(0)

            # 将标签拼接成一维向量
            labels = torch.cat(labels).to(device)

            # 输入序列长度，假设所有序列长度相同
            input_lengths = torch.full(
                size=(batch_size,), fill_value=(images.size(-1) // 4 - 1), dtype=torch.long).to(device)
            label_lengths = torch.tensor(
                label_lengths, dtype=torch.long).to(device)

            # 前向传播
            preds = model(images)
            # 计算 CTC 损失
            loss = criterion(preds, labels, input_lengths, label_lengths)
            loss_sum += loss.item()
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_progress.set_postfix(loss=f'{loss.item():.4f}')

        test_loss, test_accuracy = evaluate_model(
            test_dir, model, captcha_length, class_num, padding_index, width, height, characters)
        train_loss, train_accuracy = loss_sum / len(train_loader), acc_sum / \
            (len(train_dataset))

        if log:
            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'epoch_time': int(time.time() - start_time)
            })

        early_stopping(test_loss)
        epoch_progress.set_postfix(
            loss=f'{train_loss:.4f}', accuracy=f'{100 * train_accuracy:.4f}%')

        if epoch % 100 == 0:
            torch.save(model.state_dict(), model_path.replace(
                'model.pth', f'model_{epoch}.pth'))
        if early_stopping.early_stop:
            print('Early stopping in epoch:', epoch)
            break

    torch.save(model.state_dict(), model_path)
    if log:
        wandb.finish()


if __name__ == '__main__':
    config = load_config()
    training_config = config['training']
    testing_config = config['testing']
    dataset_config = config['dataset']
    model_config = config['model']
    train(train_dir=training_config['train_dir'], test_dir=testing_config['test_dir'], characters=dataset_config['characters'], batch_size=64, pretrained=False,
          epochs=10, captcha_length=dataset_config['captcha_length'], class_num=37, padding_index="36",  model_path='./model/model-test.pth', learning_rate=0.001, width=model_config['width'], height=model_config['height'], log=False, early_stopping={})
