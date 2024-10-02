from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import TrainingArguments
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertForSequenceClassification
from transformers import BertTokenizer, BertConfig


from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from dataset import NewsDataset


# 1. 加载数据集
newsgroups = fetch_20newsgroups(
    subset='all', remove=('headers', 'footers', 'quotes'))

texts = newsgroups.data   # type: ignore
labels = newsgroups.target  # type: ignore
target_names = newsgroups.target_names  # type: ignore

# 3. 数据分割
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

num_labels = np.max(labels) + 1
model_name = 'bert-base-uncased'
id2label = {i: name for i, name in enumerate(target_names)}
label2id = {name: i for i, name in enumerate(target_names)}
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(
    model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)
model = BertForSequenceClassification.from_pretrained(
    model_name, config=config)

train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
val_dataset = NewsDataset(val_texts, val_labels, tokenizer)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# 学习率调度器
training_args = TrainingArguments(
    output_dir='./results',          # 模型输出目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=16,  # 训练时每个设备的批次大小
    per_device_eval_batch_size=16,   # 验证时每个设备的批次大小
    warmup_steps=500,                # 预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,                # 日志记录步数
    eval_strategy="steps",     # 评估策略（每多少步评估一次）
    eval_steps=500,                  # 评估步数
    save_steps=1000,                 # 模型保存步数
    load_best_model_at_end=True      # 在训练结束时加载最佳模型
)
total_steps = len(train_loader) * training_args.num_train_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=total_steps
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # type: ignore

for epoch in range(int(training_args.num_train_epochs)):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} average loss: {avg_train_loss:.4f}")

    # 在验证集上评估
    model.eval()
    val_accuracy = 0
    val_steps = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    # 计算准确率
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")

# 保存模型
model.save_pretrained('./saved_model')
