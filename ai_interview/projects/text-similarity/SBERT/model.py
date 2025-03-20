import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# 1. 定义 Siamese BERT 模型


class SiameseBERT(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', dropout=0.1):
        super(SiameseBERT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        # BERT 的 hidden_size 通常是 768
        self.fc = nn.Linear(768, 128)  # 输出降维到 128 维嵌入

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        # 取 [CLS] token 的输出作为文档表示
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        embedding = self.fc(pooled_output)
        return embedding

# 2. 定义数据集


class SiameseDataset(Dataset):
    def __init__(self, doc_pairs, labels, tokenizer, max_length=128):
        """
        doc_pairs: List of tuples (doc1, doc2)
        labels: 1 for similar, 0 for dissimilar
        """
        self.doc_pairs = doc_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.doc_pairs)

    def __getitem__(self, idx):
        doc1, doc2 = self.doc_pairs[idx]
        label = self.labels[idx]

        # 对两个文档进行 tokenization
        encoding1 = self.tokenizer(doc1,
                                   max_length=self.max_length,
                                   padding='max_length',
                                   truncation=True,
                                   return_tensors='pt')

        encoding2 = self.tokenizer(doc2,
                                   max_length=self.max_length,
                                   padding='max_length',
                                   truncation=True,
                                   return_tensors='pt')

        return {
            'input_ids_1': encoding1['input_ids'].squeeze(0),
            'attention_mask_1': encoding1['attention_mask'].squeeze(0),
            'input_ids_2': encoding2['input_ids'].squeeze(0),
            'attention_mask_2': encoding2['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }

# 3. 对比损失函数


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # 计算欧几里得距离
        euclidean_distance = torch.nn.functional.pairwise_distance(
            output1, output2)
        # 对比损失
        loss_similar = label * torch.pow(euclidean_distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = torch.mean(loss_similar + loss_dissimilar) / 2
        return loss

# 4. 训练函数


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            # 将数据移到设备上
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # 前向传播
            output1 = model(input_ids_1, attention_mask_1)
            output2 = model(input_ids_2, attention_mask_2)

            # 计算损失
            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 5. 主函数示例


def main():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化 tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = SiameseBERT(bert_model_name='bert-base-chinese').to(device)

    # 示例数据
    doc_pairs = [
        ("这是一篇关于人工智能的文章", "人工智能技术正在快速发展"),
        ("我喜欢吃中餐", "今天天气很好"),
    ]
    labels = [1, 0]  # 1 表示相似，0 表示不相似

    # 创建数据集和数据加载器
    dataset = SiameseDataset(doc_pairs, labels, tokenizer)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 定义损失函数和优化器
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # 训练模型
    train_model(model, train_loader, criterion,
                optimizer, num_epochs=5, device=device)

    # 推理示例
    model.eval()
    with torch.no_grad():
        test_doc1 = "人工智能很强大"
        test_doc2 = "AI技术很有前景"
        encoding1 = tokenizer(test_doc1, return_tensors='pt', max_length=128,
                              padding='max_length', truncation=True).to(device)
        encoding2 = tokenizer(test_doc2, return_tensors='pt', max_length=128,
                              padding='max_length', truncation=True).to(device)

        emb1 = model(**encoding1)
        emb2 = model(**encoding2)
        distance = torch.nn.functional.pairwise_distance(emb1, emb2)
        print(f"Distance between documents: {distance.item():.4f}")


if __name__ == "__main__":
    main()
