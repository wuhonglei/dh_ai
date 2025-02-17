import torch
import numpy as np
from torch.utils.data import DataLoader
from pprint import pprint
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import NamesDataset
from model import RNNModel

from typing import Any

dataset = NamesDataset('data/names')
# 定义划分比例
train_size = int(0.9 * len(dataset))  # 80% 作为训练集
val_size = len(dataset) - train_size  # 20% 作为验证集
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model_name = 'name_classify.pth'
input_size = len(dataset.all_letters)
hidden_size = 128
output_size = dataset.get_labels_num()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(input_size, hidden_size, output_size)
model.to(device)
model.load_state_dict(torch.load(model_name))

model.eval()
total = 0
correct = 0
confusion = np.zeros((output_size, output_size), dtype=int)
y_true = []
y_pred = []
with torch.no_grad():
    for name, label in dataloader:
        hidden = model.init_hidden()
        name_tensor = dataset.name_to_tensor(name[0])
        name_tensor = name_tensor.to(device)
        label = label.to(device)
        for j in range(name_tensor.size(0)):
            hidden = model(name_tensor[j], hidden)

        output = model.compute_output(hidden)
        _, predicted = torch.max(output, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        confusion[label.item()][predicted.item()] += 1
        y_true.append(label.item())
        y_pred.append(predicted.item())

print(f'accuracy: {correct / total}')
cm = confusion_matrix(y_true, y_pred, labels=range(
    output_size))

report = classification_report(
    y_true, y_pred, labels=range(
        output_size), target_names=dataset.get_label_names(), zero_division='warn')

print('report', report)

# 可视化混淆矩阵
sns.set()  # 使用默认样式
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
ax.set_xlabel('Predicted Labels')
ax.set_title('Confusion Matrix')
plt.show()
