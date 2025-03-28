import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
# TextCNN 模型定义


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, filter_sizes: list[int], num_filters: int, dropout: float = 0.5, pretrained_embeddings: Optional[np.ndarray] = None):
        """
        参数说明:
        - vocab_size: 词表大小
        - embed_dim: 词嵌入维度
        - num_classes: 分类类别数
        - filter_sizes: 卷积核高度列表（如 [2, 3, 4]）
        - num_filters: 每种卷积核的数量
        - dropout: dropout 比例
        - pretrained_embeddings: 预训练词向量（可选，形状为 [vocab_size, embed_dim]）
        """
        super(TextCNN, self).__init__()

        # 词嵌入层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings, dtype=torch.float), freeze=False)  # freeze=False 允许微调
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 卷积层列表，每种 filter_size 一个卷积核
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters,
                      kernel_size=(fs, embed_dim))
            for fs in filter_sizes
        ])

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        """
        前向传播
        - x: 输入张量，形状为 [batch_size, max_seq_len]
        """
        # 词嵌入，形状变为 [batch_size, max_seq_len, embed_dim]
        x = self.embedding(x)

        # 增加通道维度，变为 [batch_size, 1, max_seq_len, embed_dim]
        x = x.unsqueeze(1)

        # 卷积和池化
        conv_outputs = []
        for conv in self.convs:
            # 卷积，输出形状 [batch_size, num_filters, conv_out_len, 1]
            conv_out = F.relu(conv(x))
            # 最大池化，输出形状 [batch_size, num_filters]
            pool_out = F.max_pool2d(
                conv_out, (conv_out.size(2), 1)).squeeze(3).squeeze(2)
            conv_outputs.append(pool_out)

        # 拼接所有卷积核的输出，形状 [batch_size, len(filter_sizes) * num_filters]
        x = torch.cat(conv_outputs, dim=1)

        # Dropout 和全连接层
        x = self.dropout(x)
        logits = self.fc(x)

        return logits

# 自定义数据集类（示例）


class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=50):
        self.texts = texts  # 分词后的文本列表
        self.labels = labels  # 标签列表
        self.word2idx = word2idx  # 词到索引的映射
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 转换为索引序列，填充或截断到 max_len
        seq = [self.word2idx.get(word, 0)
               for word in text][:self.max_len]  # 0 表示未知词
        seq = seq + [0] * (self.max_len - len(seq)
                           ) if len(seq) < self.max_len else seq
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 训练函数


def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_acc = 0, 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_acc += (preds == labels).float().mean().item()

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {total_acc/len(train_loader):.4f}")

        # 验证
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                logits = model(texts)
                preds = torch.argmax(logits, dim=1)
                val_acc += (preds == labels).float().mean().item()
        print(f"Val Acc: {val_acc/len(val_loader):.4f}")


# 示例用法
if __name__ == "__main__":
    # 超参数
    VOCAB_SIZE = 5000  # 词表大小
    EMBED_DIM = 100    # 词嵌入维度
    NUM_CLASSES = 2    # 二分类任务
    FILTER_SIZES = [2, 3, 4]  # 卷积核高度
    NUM_FILTERS = 100  # 每种卷积核数量
    DROPOUT = 0.5
    MAX_LEN = 50       # 最大序列长度
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟数据
    texts = [["我", "很", "喜欢", "这", "本书"], ["这", "电影", "不好"]]  # 分词后的文本
    labels = [1, 0]  # 正面、负面
    word2idx = {"<PAD>": 0, "我": 1, "很": 2, "喜欢": 3,
                "这": 4, "本书": 5, "电影": 6, "不好": 7}  # 词表

    # 数据加载
    dataset = TextDataset(texts, labels, word2idx, MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)  # 这里用相同数据仅作示例

    # 初始化模型（无预训练词向量）
    model = TextCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        filter_sizes=FILTER_SIZES,
        num_filters=NUM_FILTERS,
        dropout=DROPOUT
    )

    # 如果有预训练词向量，可以这样加载（假设 pretrained_embeddings 是 numpy 数组）
    # pretrained_embeddings = np.random.rand(VOCAB_SIZE, EMBED_DIM)  # 模拟预训练词向量
    # model = TextCNN(
    #     vocab_size=VOCAB_SIZE,
    #     embed_dim=EMBED_DIM,
    #     num_classes=NUM_CLASSES,
    #     filter_sizes=FILTER_SIZES,
    #     num_filters=NUM_FILTERS,
    #     dropout=DROPOUT,
    #     pretrained_embeddings=pretrained_embeddings
    # )

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    train_model(model, train_loader, val_loader,
                NUM_EPOCHS, criterion, optimizer, DEVICE)
