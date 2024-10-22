from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
from tqdm import tqdm

from dataset import collate_batch
from dataset import get_vocab
from dataset import KeywordCategoriesDataset
# from models.rnn_model import KeywordCategoryModel
# from models.simple_model import KeywordCategoryModel
from models.category_model import KeywordCategoryModel
from utils.model import save_training_json


def train(train_keywords: list[str], train_labels: list[str], country: str, test_keywords: list[str], test_labels: list[str]):
    train_dataset = KeywordCategoriesDataset(
        train_keywords, train_labels, country, use_cache=True)
    test_dataset = KeywordCategoriesDataset(
        test_keywords, test_labels, country, use_cache=True)

    # 小批量读取数据
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1024,
                                  shuffle=True,
                                  )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,)
    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    # 定义模型的必要参数
    hidden_size = [
        64,
        31
    ]
    dropout = 0
    num_classes = len(train_dataset.label2index)
    input_size = train_dataset[0][0].shape[0]
    num_epochs = 1000
    learning_rate = 0.01

    # 定义模型
    model = KeywordCategoryModel(
        input_size, hidden_size, num_classes, dropout)
    # model.load_state_dict(torch.load(
    #     f"./models/weights/{country}_model.pth", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    epoch_progress = tqdm(range(num_epochs), leave=True)
    for epoch in epoch_progress:
        epoch_progress.set_description(f'epoch: {epoch + 1}')

        model.train()
        batch_progress = tqdm(enumerate(train_dataloader), leave=False)
        for batch_idx, (x, y) in batch_progress:
            batch_progress.set_description(
                f'batch: {batch_idx + 1}/{len(train_dataloader)}')

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            predict = model(x)
            loss = criterion(predict, y)
            loss.backward()
            optimizer.step()
            batch_progress.set_postfix(loss=loss.item())

        train_acc = evaluate(train_dataloader, model)
        test_acc = evaluate(test_dataloader, model)
        epoch_progress.set_postfix(train_acc=train_acc, test_acc=test_acc)

    # 保存模型
    torch.save(model.state_dict(), f"./models/weights/{country}_model.pth")


def evaluate(dataloader: DataLoader, model):
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for text, label in dataloader:
            text = text.to(DEVICE)
            label = label.to(DEVICE)
            predict = model(text)
            _, predicted = torch.max(predict.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    # print(f"Accuracy: {correct / total * 100:.2f}%")
    return f'{correct / total * 100:.2f}%'
