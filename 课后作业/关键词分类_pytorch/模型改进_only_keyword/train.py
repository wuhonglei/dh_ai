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


def train(X: Series, sub_category: Series, y: Series, country: str, ):
    dataset = KeywordCategoriesDataset(
        X.tolist(), sub_category, y.tolist(), country, use_cache=False)

    # 使用 train_test_split 将数据划分为训练集和测试集
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.05, random_state=42)
    min_freq = 10
    vocab, vocab_cache = get_vocab(
        train_dataset, country, min_freq, use_cache=False)

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
    train_args = {
        "vocab_size": len(vocab),
        "embed_dim": len(vocab) // 2,
        "hidden_size": 128,
        "num_classes": len(dataset.label2index),
        "padding_idx": vocab['<PAD>'],
        "num_epochs": 20,
        "min_freq": min_freq,
        "learning_rate": 0.01,  # type: ignore
        'batch_size': 2048,
        'vocab_cache': vocab_cache,
        'sub_category': len(dataset[0][1]),
        'save_checkpoint': False,
        # 'load_state_dict': f"./models/weights/{country}/TH_LSTM_128*2_fc_2_seo_1731669164_final.pth",
        'save_model': f'{country}_LSTM_128*2_fc_2_seo_{unix_time}',
        'log_file': f"./logs/{country}/{country}_{unix_time}.txt"
    }
    save_training_json(train_args, f"./config/params/{country}_params.json")

    # 定义模型
    model = KeywordCategoryModel(
        train_args['vocab_size'], train_args['embed_dim'], train_args['hidden_size'], train_args['sub_category'], train_args['num_classes'], train_args['padding_idx'])
    if train_args.get('load_state_dict'):
        # init_model(
        #     model, f"./models/weights/SG/SG_LSTM_128*2_fc_2_bpv_model.pth", DEVICE)
        model.load_state_dict(torch.load(
            train_args['load_state_dict'], map_location=DEVICE, weights_only=True))
        pass
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=train_args['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    epoch_progress = tqdm(range(train_args['num_epochs']), leave=True)
    for epoch in epoch_progress:
        epoch_progress.set_description(f'epoch: {epoch + 1}')

        model.train()
        loss_sum = 0.0
        batch_progress = tqdm(enumerate(train_dataloader), leave=False)
        for batch_idx, (text, sub_category, label) in batch_progress:
            batch_progress.set_description(
                f'batch: {batch_idx + 1}/{len(train_dataloader)}')

            text = text.to(DEVICE)
            sub_category = sub_category.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            predict = model(text, sub_category)
            loss = criterion(predict, label)
            loss_sum += loss
            if (batch_idx + 1) % train_args['batch_size'] == 0:
                loss_sum.backward()
                # 防止梯度爆炸
                # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_avg = loss_sum.item() / train_args['batch_size']
                batch_progress.set_postfix(loss=loss_avg)
                loss_sum = 0.0

        if loss_sum != 0.0:
            loss_sum.backward()  # type: ignore
            # 防止梯度爆炸
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum = 0.0

        train_acc = 'empty'
        if len(dataset) > 20000 and (epoch + 1) % 2 == 0:
            train_acc = evaluate(train_dataloader, model)
        if len(dataset) > 100000 and (epoch + 1) % 4 == 0:
            train_acc = evaluate(train_dataloader, model)
        if len(dataset) < 20000:
            train_acc = evaluate(train_dataloader, model)
        test_acc = evaluate(test_dataloader, model)
        desc = f'epcoh: {epoch + 1}; test acc: {test_acc}; train acc: {train_acc}; loss_avg: {loss_avg}'
        write_to_file(train_args['log_file'],
                      time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '; ' + desc)
        epoch_progress.set_postfix(
            test_acc=test_acc, train_acc=train_acc, loss_avg=loss_avg)
        if train_args.get('save_checkpoint') and (epoch + 1) % 3 == 0 and train_args.get('save_model'):
            # 保存模型
            torch.save(model.state_dict(),
                       f"./models/weights/{country}/{train_args['save_model']}_{epoch + 1}.pth")

    if train_args.get('save_model'):
        # 保存模型
        torch.save(model.state_dict(
        ), f"./models/weights/{country}/{train_args['save_model']}_final.pth")


def evaluate(dataloader: DataLoader, model):
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for text, sub_category, label in dataloader:
            text = text.to(DEVICE)
            label = label.to(DEVICE)
            sub_category = sub_category.to(DEVICE)
            predict = model(text, sub_category)
            _, predicted = torch.max(predict.data, 1)
            y_true.extend(label.tolist())
            y_pred.extend(predicted.tolist())
    accuracy = accuracy_score(y_true, y_pred)
    return f'{accuracy * 100:.2f}%'
