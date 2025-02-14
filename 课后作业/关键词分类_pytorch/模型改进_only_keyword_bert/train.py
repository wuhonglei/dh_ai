from pandas import Series
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

from dataset import KeywordCategoriesDataset
from models.simple_model import KeywordCategoryModel
from utils.model import save_training_json, get_class_weights

# bert 中文模型
bert_name = 'bert-base-uncased'


def train(X: Series, y: Series, country: str, ):
    dataset = KeywordCategoriesDataset(bert_name,
                                       X.tolist(), y.tolist(), country, use_cache=True, use_config=True)

    # 使用 train_test_split 将数据划分为训练集和测试集
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.05, random_state=42)

    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    hidden_size = 256
    num_classes = dataset.num_classes
    num_epochs = 3
    learning_rate = 2e-5
    eps = 1e-8
    batch_size = 128
    dropout = 0.1

    save_training_json({
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,  # type: ignore
        'batch_size': batch_size,
        'dropout': dropout
    }, f"./config/{country}_params.json")

    # 小批量读取数据
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,)

    # 定义模型
    model = KeywordCategoryModel(
        bert_name, hidden_size, num_classes, dropout)
    # model.load_state_dict(torch.load(
    #     f"./models/weights/SG_model_shopee_epoch_5.pth", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    # 定义优化器参数，通常只优化需要训练的参数
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),  # type: ignore
                            lr=learning_rate,  # 学习率
                            eps=eps          # 稳定性参数
                            )
    # 定义学习率调度器
    # 计算总训练步数
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,               # 预热步数
                                                num_training_steps=total_steps)    # 总训练步数
    criterion = nn.CrossEntropyLoss(
        # weight=get_class_weights(y_train).to(DEVICE),
    )

    epoch_progress = tqdm(range(num_epochs), leave=True)
    for epoch in epoch_progress:
        epoch_progress.set_description(f'epoch: {epoch + 1}')

        model.train()
        total_loss = 0.0
        batch_progress = tqdm(enumerate(train_dataloader), leave=False)
        for batch_idx, (input_ids, attention_mask, labels) in batch_progress:
            batch_progress.set_description(
                f'batch: {batch_idx + 1}/{len(train_dataloader)}')

            # 获取批次数据并移动到设备
            b_input_ids = input_ids.to(DEVICE)
            b_attention_mask = attention_mask.to(DEVICE)
            b_labels = labels.to(DEVICE)

            optimizer.zero_grad()
            predict = model(b_input_ids, b_attention_mask)
            loss = criterion(predict, b_labels)
            total_loss += loss.item()
            loss.backward()

            # 防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # 计算该轮的平均训练损失
        avg_train_loss = total_loss / len(train_dataloader)
        test_acc = evaluate(test_dataloader, model)
        train_acc = 'unknown'
        if (epoch + 1) % 3 == 0:
            train_acc = evaluate(train_dataloader, model)
        epoch_progress.set_postfix(
            avg_train_loss=avg_train_loss, train_acc=train_acc, test_acc=test_acc)

    # 保存模型
    torch.save(model.state_dict(), f"./models/weights/{country}_model.pth")


def evaluate(dataloader: DataLoader, model):
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for (input_ids, attention_mask, labels) in dataloader:
            # 获取批次数据并移动到设备
            b_input_ids = input_ids.to(DEVICE)
            b_attention_mask = attention_mask.to(DEVICE)
            b_labels = labels.to(DEVICE)

            predict = model(
                b_input_ids, b_attention_mask
            )
            _, predicted = torch.max(predict.data, 1)
            y_true.extend(b_labels.tolist())
            y_pred.extend(predicted.tolist())
    accuracy = accuracy_score(y_true, y_pred)
    return f'{accuracy * 100:.2f}%'
