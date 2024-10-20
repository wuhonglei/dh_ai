from typing import Union
from pandas import DataFrame
from model import KeywordCategoryModel
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
    dataset = KeywordCategoriesDataset(train_keywords, train_labels, country)
    vocab = build_vocab(dataset)
    # 词表的保存
    with open("./vocab/tw_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # 回调函数，用于不同长度的文本进行填充
    def collate(batch): return collate_batch(batch, vocab)
    # 小批量读取数据
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            collate_fn=collate)
    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    # 定义模型的必要参数
    vocab_size = len(vocab)
    embed_dim = 128
    hidden_size = 128
    num_classes = len(dataset.label2index)
    padding_idx = vocab["<pad>"]
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 1024

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
        batch_progress = tqdm(enumerate(dataloader), leave=False)
        for batch_idx, (text, label) in batch_progress:
            batch_progress.set_description(
                f'batch: {batch_idx + 1}/{len(dataloader)}')

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

        acc = evaluate(test_keywords, test_labels, country, model)
        epoch_progress.set_postfix(acc=acc)

    # 保存模型
    torch.save(model.state_dict(), f"./models/{country}_model.pth")


def evaluate(keywords: list[str], labels: list[str], country: str, model):
    dataset = KeywordCategoriesDataset(keywords, labels, country)
    with open("./vocab/tw_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    def collate(batch): return collate_batch(batch, vocab)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate)
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')

    # train_params = load_training_json(f"./config/{country}_params.json")

    # model = KeywordCategoryModel(
    #     train_params['vocab_size'], train_params['embed_dim'], train_params['num_classes'], train_params['padding_idx'])
    # model.load_state_dict(torch.load(
    #     f"./models/{country}_model.pth", map_location=DEVICE, weights_only=True))
    # model.to(DEVICE)
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
