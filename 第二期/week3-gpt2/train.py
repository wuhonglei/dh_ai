import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from gpt_1 import GPT1
from transformers import GPT2Tokenizer


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, max_length=max_length, padding=True, truncation=True)
        self.labels = self.encodings['input_ids'].clone()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def train_model(model, train_loader, optimizer, device, num_epochs, max_grad_norm=1.0):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch in progress_bar:
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(input_ids)

            # 计算损失
            # 将输出展平以匹配标签形状
            outputs = outputs.view(-1, outputs.size(-1))
            labels = labels.view(-1)

            # 忽略填充标记的损失
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # 更新参数
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 模型参数
    vocab_size = 10000
    d_model = 768
    num_heads = 12
    num_layers = 12
    d_ff = 3072
    max_seq_length = 512
    dropout = 0.1

    # 训练参数
    batch_size = 8
    learning_rate = 3e-5
    num_epochs = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT1(vocab_size, d_model, num_heads,
                 num_layers, d_ff, max_seq_length, dropout)
    model.to(device)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # 这里需要替换为您的实际数据
    # 示例：假设您有一个文本列表
    texts = ["示例文本1", "示例文本2"]

    # 创建数据集和数据加载器
    dataset = TextDataset(texts, tokenizer, max_seq_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练模型
    train_model(model, train_loader, optimizer, device, num_epochs)

    # 保存模型
    torch.save(model.state_dict(), 'gpt1_model.pth')


if __name__ == "__main__":
    main()
