import torch
import torch.nn as nn


def train(model, img, label, model_name):
    label = torch.tensor([label], dtype=torch.int64).view(1)
    """ 对于识别错误的图片，使用正确的标签进行训练 """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    torch.save(model.state_dict(), f'./models/{model_name}')
    model.eval()
