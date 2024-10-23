from pandas import Series
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup

from dataset import KeywordCategoriesDataset
from models.simple_model import KeywordCategoryModel
from utils.model import save_training_json, get_class_weights


def train(X: Series, y: Series, country: str, ):
    # 使用 train_test_split 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X.tolist(), y.tolist(), test_size=0.05, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.05, random_state=0)

    train_dataset = KeywordCategoriesDataset(
        X_train, y_train, country, use_cache=True)
    test_dataset = KeywordCategoriesDataset(
        X_test, y_test, country, use_cache=True)
    val_dataset = KeywordCategoriesDataset(
        X_val, y_val, country, use_cache=True)

    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    hidden_size = 256
    num_classes = len(train_dataset.label_encoder.classes_)
    num_epochs = 5
    learning_rate = 2e-5
    eps = 1e-8
    batch_size = 32
    dropout = 0.25

    save_training_json({
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,  # type: ignore
        'batch_size': batch_size,
    }, f"./config/{country}_params.json")

    # 小批量读取数据
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,)

    # 定义模型
    model = KeywordCategoryModel(
        'bert-base-uncased', hidden_size, num_classes, dropout)
    # model.load_state_dict(torch.load(
    #     f"./models/weights/{country}_model.pth", map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    # 定义优化器参数，通常只优化需要训练的参数
    optimizer = optim.AdamW(model.parameters(),
                            lr=learning_rate,          # 学习率
                            eps=eps          # 稳定性参数
                            )
    # 定义学习率调度器
    # 计算总训练步数
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,               # 预热步数
                                                num_training_steps=total_steps)    # 总训练步数
    criterion = nn.CrossEntropyLoss(
        weight=get_class_weights(y_train).to(DEVICE)
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
        val_acc = evaluate(val_dataloader, model)

        epoch_progress.set_postfix(
            avg_train_loss=avg_train_loss, val_acc=val_acc)

    test_acc = evaluate(test_dataloader, model)
    epoch_progress.set_postfix(
        avg_train_loss=avg_train_loss, val_acc=val_acc, test_acc=test_acc)

    # 保存模型
    torch.save(model.state_dict(), f"./models/weights/{country}_model.pth")


def evaluate(dataloader: DataLoader, model):
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    model.eval()
    correct = 0
    total = 0
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
            total += labels.size(0)
            correct += (predicted == b_labels).sum().item()
    # print(f"Accuracy: {correct / total * 100:.2f}%")
    return f'{correct / total * 100:.2f}%'
