import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CNNModel
from dataset import CaptchaDataset


def evaluate(data_dir, model_path, captcha_length: int, class_num, padding_index):
    model = CNNModel(captcha_length, class_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True))
    model.to(device)
    return evaluate_model(data_dir, model, captcha_length, class_num, padding_index)


def evaluate_model(data_dir, model, captcha_length, class_num, padding_index):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    loader_config = {'num_workers': 4, 'pin_memory': True} if is_cuda else {}
    eval_dataset = CaptchaDataset(
        data_dir, captcha_length=captcha_length, padding_index=padding_index, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=1, **loader_config)

    loss_sum = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    model.eval()
    for imgs, labels in eval_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            output = model(imgs)

        predict = output.argmax(dim=2, keepdim=True)
        correct += (predict == labels.view_as(predict)).all(dim=1).sum().item()
        total += labels.size(0)

        # output: (batch_size, captcha_length, class_num)
        output = output.view(-1, class_num)
        # (batch_size * captcha_length)
        labels = labels.view(-1)

        loss = criterion(output, labels)
        loss_sum += loss.item() * imgs.size(0)

    test_loss = loss_sum / total
    test_accuracy = 1.0 * correct / (total)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {100 * test_accuracy:.4f}%')

    return test_loss, test_accuracy


if __name__ == '__main__':
    evaluate(data_dir='./data/test_acc', model_path='./models/2-model.pth',
             captcha_length=2, class_num=2, padding_index='10')
