import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm

# 1. 数据集类


class ProductTitleDataset(Dataset):
    def __init__(self, titles, level1_labels, leaf_labels, tokenizer, max_len=64):
        self.titles = titles
        self.level1_labels = level1_labels
        self.leaf_labels = leaf_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = str(self.titles[idx])
        level1 = self.level1_labels[idx]
        leaf = self.leaf_labels[idx]

        encoding = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'level1': torch.tensor(level1, dtype=torch.long),
            'leaf': torch.tensor(leaf, dtype=torch.long)
        }

# 2. 级联模型定义


class CascadeBERTClassifier(nn.Module):
    def __init__(self, bert_model, num_level1, num_leaf, hidden_size=768):
        super(CascadeBERTClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        # 一级目录分类头
        self.level1_classifier = nn.Linear(hidden_size, num_level1)
        # 叶子目录分类头（输入包括 BERT 输出和一级目录嵌入）
        self.leaf_classifier = nn.Linear(hidden_size + num_level1, num_leaf)
        self.num_level1 = num_level1

    def forward(self, input_ids, attention_mask, level1_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token 的表示
        pooled_output = self.dropout(pooled_output)

        # 一级目录预测
        level1_logits = self.level1_classifier(pooled_output)

        # 叶子目录预测
        if level1_labels is not None:
            # 训练时使用真实的一级标签
            level1_one_hot = nn.functional.one_hot(
                level1_labels, num_classes=self.num_level1).float()
        else:
            # 推理时使用预测的一级标签
            level1_probs = nn.functional.softmax(level1_logits, dim=-1)
            level1_one_hot = level1_probs

        # 拼接 BERT 输出和一级目录表示
        leaf_input = torch.cat([pooled_output, level1_one_hot], dim=-1)
        leaf_logits = self.leaf_classifier(leaf_input)

        return level1_logits, leaf_logits

# 3. 训练函数


def train_model(model, data_loader, optimizer, device, num_level1, num_leaf):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        level1_labels = batch['level1'].to(device)
        leaf_labels = batch['leaf'].to(device)

        optimizer.zero_grad()
        level1_logits, leaf_logits = model(
            input_ids, attention_mask, level1_labels)

        # 计算损失
        loss_level1 = nn.CrossEntropyLoss()(level1_logits, level1_labels)
        loss_leaf = nn.CrossEntropyLoss()(leaf_logits, leaf_labels)
        loss = loss_level1 + loss_leaf

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

# 4. 验证函数


def evaluate_model(model, data_loader, device):
    model.eval()
    level1_correct, leaf_correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            level1_labels = batch['level1'].to(device)
            leaf_labels = batch['leaf'].to(device)

            level1_logits, leaf_logits = model(input_ids, attention_mask)

            level1_preds = torch.argmax(level1_logits, dim=1)
            leaf_preds = torch.argmax(leaf_logits, dim=1)

            level1_correct += (level1_preds == level1_labels).sum().item()
            leaf_correct += (leaf_preds == leaf_labels).sum().item()
            total += level1_labels.size(0)

    return level1_correct / total, leaf_correct / total

# 5. 主程序


def main():
    # 超参数
    MAX_LEN = 64
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据（假设 CSV 文件）
    df = pd.read_csv('product_data.csv')  # 包含 title, level1, leaf 列
    titles = df['title'].values
    level1_labels = df['level1'].values
    leaf_labels = df['leaf'].values

    # 标签编码
    level1_encoder = LabelEncoder()
    leaf_encoder = LabelEncoder()
    level1_labels = level1_encoder.fit_transform(level1_labels)
    leaf_labels = leaf_encoder.fit_transform(leaf_labels)
    num_level1 = len(level1_encoder.classes_)
    num_leaf = len(leaf_encoder.classes_)

    # 划分数据集
    train_titles, val_titles, train_level1, val_level1, train_leaf, val_leaf = train_test_split(
        titles, level1_labels, leaf_labels, test_size=0.2, random_state=42
    )

    # 加载 BERT 和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert_model = BertModel.from_pretrained('bert-base-chinese')

    # 创建数据集和数据加载器
    train_dataset = ProductTitleDataset(
        train_titles, train_level1, train_leaf, tokenizer, MAX_LEN)
    val_dataset = ProductTitleDataset(
        val_titles, val_level1, val_leaf, tokenizer, MAX_LEN)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 初始化模型
    model = CascadeBERTClassifier(bert_model, num_level1, num_leaf).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        avg_loss = train_model(model, train_loader,
                               optimizer, DEVICE, num_level1, num_leaf)
        level1_acc, leaf_acc = evaluate_model(model, val_loader, DEVICE)
        print(
            f'Loss: {avg_loss:.4f}, Level1 Acc: {level1_acc:.4f}, Leaf Acc: {leaf_acc:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'cascade_bert_model.pth')
    # 保存标签编码器
    import pickle
    with open('level1_encoder.pkl', 'wb') as f:
        pickle.dump(level1_encoder, f)
    with open('leaf_encoder.pkl', 'wb') as f:
        pickle.dump(leaf_encoder, f)


if __name__ == '__main__':
    main()
