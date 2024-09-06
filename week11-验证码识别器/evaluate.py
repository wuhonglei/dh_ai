from model import CNNModel
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CaptchaDataset


def evaluate(data_dir, model_path, captcha_length: int):
    model = CNNModel(captcha_length * 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return evaluate_model(data_dir, model, captcha_length)


def evaluate_model(data_dir, model, captcha_length):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    eval_dataset = CaptchaDataset(data_dir, transform=transform)
    eval_loader = DataLoader(eval_dataset, batch_size=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss_sum = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for imgs, labels in eval_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            output = model(imgs)

        loss = torch.tensor(0.0).to(device)
        for i in range(captcha_length):
            loss += criterion(output[:, i, :], labels[:, i])
        loss_sum += loss.item() * imgs.size(0)
        predict = output.argmax(dim=2, keepdim=True)
        total += labels.size(0)
        correct += (predict == labels.view_as(predict)).sum().item()

    test_loss = loss_sum / total
    test_accuracy = 1.0 * correct / total

    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {100 * test_accuracy}%')

    return test_loss, test_accuracy


if __name__ == '__main__':
    evaluate(data_dir='./data/test', model_path='./model/model.pth')
