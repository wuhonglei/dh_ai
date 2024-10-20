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


def train(keywords: list[str], labels: list[str], country: str):
    dataset = KeywordCategoriesDataset(keywords, labels, country)
    vocab = build_vocab(dataset)
    # 词表的保存
    with open("tw_vocab.pkl", "wb") as f:
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
    num_classes = len(dataset.label2index)
    padding_idx = vocab["<pad>"]
    num_epochs = 50

    # 定义模型
    model = KeywordCategoryModel(
        vocab_size, embed_dim, num_classes, padding_idx).to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        loss_sum = 0.0
        for batch_idx, (text, label) in enumerate(dataloader):
            text = text.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            predict = model(text)
            loss = criterion(predict, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (batch_idx) % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss_sum}")
                loss_sum = 0.0


def evaluate(keywords: list[str], labels: list[str], country: str):
    dataset = KeywordCategoriesDataset(keywords, labels, country)
    with open("tw_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    def collate(batch): return collate_batch(batch, vocab)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=collate)
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    vocab_size = len(vocab)
    embed_dim = 128
    num_classes = len(dataset.label2index)
    padding_idx = vocab["<pad>"]
    model = KeywordCategoryModel(
        vocab_size, embed_dim, num_classes, padding_idx).to(DEVICE)
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
    print(f"Accuracy: {correct / total * 100:.2f}%")
    return correct / total * 100
