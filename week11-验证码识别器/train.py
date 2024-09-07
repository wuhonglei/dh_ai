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


def train(data_dir: str, test_dir: str, batch_size: int, epochs: int, learning_rate: float, captcha_length: int, class_num: int, model_path: str, early_stopping={}):
    wandb.init(**get_wandb_config(captcha_length), job_type='train')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    train_dataset = CaptchaDataset(data_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNNModel(captcha_length, class_num)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(**early_stopping)

    for epoch in range(epochs):
        loss_sum = 0.0
        acc_sum = 0.0
        model.train()
        for batch_ids, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = torch.tensor(0.0).to(device)
            for i in range(captcha_length):
                loss += criterion(output[:, i, :], labels[:, i])

            predict = output.argmax(dim=2, keepdim=True)
            acc_sum += (predict == labels.view_as(predict)).sum().item()
            loss_sum += loss.item() * imgs.size(0)
            loss.backward()
            optimizer.step()
            if batch_ids % 10 == 0:
                print(
                    f'Epoch: {epoch+1}/{epochs} | Batch: {batch_ids}/{len(train_loader)} | Loss: {loss.item()}')

        test_loss, test_accuracy = evaluate_model(
            test_dir, model, captcha_length)
        train_loss, train_accuracy = loss_sum / \
            (len(train_dataset) * captcha_length), acc_sum / \
            (len(train_dataset) * captcha_length)

        wandb.log({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        })

        early_stopping(test_loss)
        torch.save(model.state_dict(), model_path)
        if early_stopping.early_stop:
            print('Early stopping in epoch:', epoch)
            break

    wandb.finish()


if __name__ == '__main__':
    train(data_dir='./data/train', test_dir='./data/test', batch_size=64,
          epochs=2, captcha_length=4, class_num=10, model_path='./model/model.pth', learning_rate=0.001)
