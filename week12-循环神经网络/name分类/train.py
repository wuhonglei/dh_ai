import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import NamesDataset
from model import RNNModel

dataset = NamesDataset('data/names')
# 定义划分比例
train_size = int(0.9 * len(dataset))  # 80% 作为训练集
val_size = len(dataset) - train_size  # 20% 作为验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

model_name = 'name_classify.pth'
input_size = len(dataset.all_letters)
hidden_size = 128
output_size = dataset.get_labels_num()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(input_size, hidden_size, output_size)
# model.load_state_dict(torch.load(model_name))
model.to(device)
criteria = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 30

epoch_progress = tqdm(range(epochs), leave=True)
for epoch in epoch_progress:
    epoch_progress.set_description(f'epoch: {epoch + 1}')

    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    batch_progress = tqdm(enumerate(train_dataloader), leave=False)
    model.train()
    for i, (name, label) in batch_progress:
        batch_progress.set_description(
            f'batch: {i + 1}/{len(train_dataloader)}')

        hidden = model.init_hidden()  # 初始化隐藏状态
        optimizer.zero_grad()
        name_tensor = dataset.name_to_tensor(name[0])
        name_tensor = name_tensor.to(device)
        label = label.to(device)
        for j in range(name_tensor.size(0)):
            hidden = model(name_tensor[j], hidden)

        output = model.compute_output(hidden)
        loss = criteria(output, label)
        total_loss += loss

        if i % 1000 == 0:
            total_loss.backward()
            optimizer.step()
            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for name, label in val_dataloader:
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
    epoch_progress.set_postfix(
        val_acc=correct/total, loss=total_loss.item()/len(train_dataloader))

torch.save(model.state_dict(), model_name)
