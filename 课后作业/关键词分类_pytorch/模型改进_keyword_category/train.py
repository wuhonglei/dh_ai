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


def train(train_data, train_labels: list[str], country: str, test_data, test_labels: list[str]):
    train_dataset = KeywordCategoriesDataset(
        train_data, train_labels, country, use_cache=False)
    test_dataset = KeywordCategoriesDataset(
        test_data, test_labels, country, use_cache=False)

    vocab = get_vocab(
        train_dataset, f"./vocab/{country}_vocab.pkl", use_cache=False)

    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    # 定义模型的必要参数
    vocab_size = len(vocab)
    embed_dim = 40
    sub_category_size = len(train_dataset[0][1])
    hidden_size = 64
    num_classes = len(train_dataset.label2index)
    padding_idx = vocab['<PAD>']
    num_epochs = 30
    learning_rate = 0.01
    batch_size = 1
    dropout = 0

    # 回调函数，用于不同长度的文本进行填充
    def collate(batch): return collate_batch(batch, vocab)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=100,
                                  shuffle=True,
                                  collate_fn=collate,
                                  )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=collate)

    # 定义模型
    model = KeywordCategoryModel(
        vocab_size, embed_dim, padding_idx, hidden_size, sub_category_size, num_classes, dropout)
    # model.load_state_dict(torch.load(
    #     f"./models/weights/{country}_model.pth", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    epoch_progress = tqdm(range(num_epochs), leave=True)
    for epoch in epoch_progress:
        epoch_progress.set_description(f'epoch: {epoch + 1}')

        model.train()
        loss_sum = 0.0
        batch_progress = tqdm(enumerate(train_dataloader), leave=False)
        for batch_idx, (word_input, sub_category_input, y) in batch_progress:
            batch_progress.set_description(
                f'batch: {batch_idx + 1}/{len(train_dataloader)}')

            word_input = word_input.to(DEVICE)
            sub_category_input = sub_category_input.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            predict = model(word_input, sub_category_input)
            loss = criterion(predict, y)
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
    torch.save(model.state_dict(), f"./models/weights/{country}_model.pth")


def evaluate(dataloader: DataLoader, model):
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (word_input, sub_category_input, y) in dataloader:
            word_input = word_input.to(DEVICE)
            sub_category_input = sub_category_input.to(DEVICE)
            y = y.to(DEVICE)
            predict = model(word_input, sub_category_input)
            _, predicted = torch.max(predict.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    # print(f"Accuracy: {correct / total * 100:.2f}%")
    return f'{correct / total * 100:.2f}%'
