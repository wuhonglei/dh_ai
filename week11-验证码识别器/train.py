import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb

from model import CNNModel
from dataset import CaptchaDataset
from utils import get_wandb_config, load_config
from evaluate import evaluate_model

wandb.require("core")

# 早停策略


class EarlyStopping:
    def __init__(self, enable=True, patience=5, min_delta=0.0001):
        self.enable = enable
        self.patience = patience
        self.delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if not self.enable:
            return False

        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss > self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train(data_dir: str, test_dir: str, batch_size: int, pretrained: bool, epochs: int, learning_rate: float, captcha_length: int, class_num: int, model_path: str, early_stopping={}):
    wandb.init(**get_wandb_config(captcha_length), job_type='train')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    loader_config = {'num_workers': 4, 'pin_memory': True} if is_cuda else {}
    train_dataset = CaptchaDataset(data_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_config)

    model = CNNModel(captcha_length, class_num)
    if pretrained and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(**early_stopping)

    for epoch in range(epochs):
        loss_sum = 0.0
        acc_sum = 0.0
        start_time = time.time()
        model.train()
        for batch_ids, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            predict = output.detach().argmax(dim=2, keepdim=True)
            acc_sum += (predict == labels.view_as(predict)
                        ).all(dim=1).sum().item()

            # (batch_size * captcha_length, class_num)
            output = output.view(-1, class_num)
            # (batch_size * captcha_length)
            labels = labels.view(-1)
            loss = criterion(output, labels)

            loss_sum += loss.item() * imgs.size(0)
            loss.backward()
            optimizer.step()
            if batch_ids % 10 == 0:
                print(
                    f'Epoch: {epoch+1}/{epochs} | Batch: {batch_ids}/{len(train_loader)} | Loss: {loss.item()}')

        test_loss, test_accuracy = evaluate_model(
            test_dir, model, captcha_length, class_num)
        train_loss, train_accuracy = loss_sum / \
            (len(train_dataset)), acc_sum / \
            (len(train_dataset))

        wandb.log({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'epoch_time': int(time.time() - start_time)
        })

        early_stopping(test_loss)
        torch.save(model.state_dict(), model_path)
        if early_stopping.early_stop:
            print('Early stopping in epoch:', epoch)
            break

    wandb.finish()


if __name__ == '__main__':
    train(data_dir='./data/train', test_dir='./data/test', batch_size=64, pretrained=False,
          epochs=1, captcha_length=1, class_num=10, model_path='./model/model-test.pth', learning_rate=0.001)
