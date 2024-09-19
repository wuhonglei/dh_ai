import time

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 加载模型
from model import VGG16, transform

test_dataset = datasets.MNIST(
    root='data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=256,
                         shuffle=False)

model = VGG16()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("./models/model_10.pth",
                      map_location=device, weights_only=True))

# 测试模型
model.eval()
correct = 0.0
total = 0
start_time = time.time()
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    with torch.no_grad():
        output = model(data)
    correct += (output.argmax(1) == target).sum().item()
    total += target.size(0)
    print('correct:', correct)
    break

duration = time.time() - start_time
avg_time = duration / total
print(f"Duration: {duration}, Average time: {avg_time}")
print(f"Accuracy: {correct / total}")
