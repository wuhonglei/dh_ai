from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
from tqdm import tqdm
from pandas import Series

from dataset import collate_batch
from dataset import get_vocab
from dataset import KeywordCategoriesDataset
# from models.rnn_model import KeywordCategoryModel
# from models.simple_model import KeywordCategoryModel
from models.lstm_model import KeywordCategoryModel, init_model
from utils.model import save_training_json
from utils.common import write_to_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
unix_time = int(time.time())


def train(X: Series, y: Series, country: str, ):
    dataset = KeywordCategoriesDataset(
        X.tolist(), y.tolist(), country, use_cache=True)

    # 使用 train_test_split 将数据划分为训练集和测试集
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.05, random_state=42)
    vocab = get_vocab(train_dataset, country, use_cache=True)

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
    embed_dim = 25
    hidden_size = 128
    num_classes = len(dataset.label2index)
    padding_idx = vocab['<PAD>']
    num_epochs = 35
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
    init_model(model, f"./models/weights/SG_LSTM_128*2_fc_2_bpv_model.pth", DEVICE)
    # model.load_state_dict(torch.load(
    #     f"./models/weights/SG_LSTM_128*2_fc_2_bpv_model.pth", map_location=DEVICE, weights_only=True))
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
                # 防止梯度爆炸
                # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                batch_progress.set_postfix(loss=loss_sum.item() / batch_size)
                loss_sum = 0.0

        if loss_sum != 0.0:
            loss_sum.backward()  # type: ignore
            # 防止梯度爆炸
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum = 0.0

        train_acc = evaluate(train_dataloader, model)
        test_acc = evaluate(test_dataloader, model)
        desc = f'epcoh: {epoch + 1}; test acc: {test_acc}; train acc: {train_acc}'
        write_to_file(f"./logs/{country}_{unix_time}.txt",
                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ' :' + desc)
        epoch_progress.set_postfix(test_acc=test_acc, train_acc=train_acc)

    # 保存模型
    torch.save(model.state_dict(), f"./models/weights/{country}_model.pth")


def evaluate(dataloader: DataLoader, model):
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for text, label in dataloader:
            text = text.to(DEVICE)
            label = label.to(DEVICE)
            predict = model(text)
            _, predicted = torch.max(predict.data, 1)
            y_true.extend(label.tolist())
            y_pred.extend(predicted.tolist())
    accuracy = accuracy_score(y_true, y_pred)
    return f'{accuracy * 100:.2f}%'
