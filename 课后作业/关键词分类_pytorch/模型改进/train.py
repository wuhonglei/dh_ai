from typing import Union
from pandas import DataFrame
from model import KeywordCategoryModel
# from simple_model import KeywordCategoryModel
from utils.model import get_class_weights
from dataset import collate_batch
from dataset import build_vocab
from dataset import KeywordCategoriesDataset
from torch.utils.data import DataLoader
import pickle
from torch import optim
from torch import nn
import torch
import json
from tqdm import tqdm


def save_training_json(params: dict[str, int], path: str):
    with open(path, "w") as f:
        f.write(json.dumps(params, indent=4))


def load_training_json(path: str) -> dict[str, int]:
    with open(path, "r") as f:
        return json.loads(f.read())


def train(train_keywords: list[str], train_labels: list[str], country: str, test_keywords: list[str], test_labels: list[str]):
    train_dataset = KeywordCategoriesDataset(
        train_keywords, train_labels, country)
    test_dataset = KeywordCategoriesDataset(
        test_keywords, test_labels, country)
    vocab = build_vocab(train_dataset)
    # 词表的保存
    with open(f"./vocab/{country}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # 回调函数，用于不同长度的文本进行填充
    def collate(batch): return collate_batch(batch, vocab)
    # 小批量读取数据
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  collate_fn=collate)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=collate)
    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    # 定义模型的必要参数
    vocab_size = len(vocab)
    embed_dim = 128
    hidden_size = 64
    num_classes = len(train_dataset.label2index)
    padding_idx = vocab['<PAD>']
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 2048

    save_training_json({
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "padding_idx": padding_idx,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,  # type: ignore
        'batch_size': batch_size,
    }, f"./config/{country}_params.json")

    # 定义模型
    model = KeywordCategoryModel(
        vocab_size, embed_dim, hidden_size, num_classes, padding_idx)
    # model.load_state_dict(torch.load(
    #     f"./models/{country}_model.pth", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    epoch_progress = tqdm(range(num_epochs), leave=True)
    for epoch in epoch_progress:
        epoch_progress.set_description(f'epoch: {epoch + 1}')

        model.train()
        loss_sum = 0.0
        batch_progress = tqdm(enumerate(train_dataloader), leave=False)
        for batch_idx, (text, label) in batch_progress:
            batch_progress.set_description(
                f'batch: {batch_idx + 1}/{len(train_dataloader)}')

            text = text.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            predict = model(text)
            loss = criterion(predict, label)
            loss_sum += loss
            if (batch_idx + 1) % batch_size == 0:
                loss_sum.backward()
                optimizer.step()
                batch_progress.set_postfix(loss=loss_sum.item() / batch_size)
                loss_sum = 0.0

        if loss_sum != 0.0:
            loss_sum.backward()  # type: ignore
            optimizer.step()
            loss_sum = 0.0

        train_acc = evaluate(train_dataloader, model)
        test_acc = evaluate(test_dataloader, model)
        epoch_progress.set_postfix(train_acc=train_acc, test_acc=test_acc)

    # 保存模型
    torch.save(model.state_dict(), f"./models/{country}_model.pth")


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
