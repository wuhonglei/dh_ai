import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

# 自定义数据集


class ProductDataset(Dataset):
    def __init__(self, titles, images, labels, tokenizer, transform):
        self.titles = titles
        self.images = images
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 文本处理
        title = self.titles[idx]
        encoding = self.tokenizer(
            title, return_tensors='pt', max_length=32, truncation=True, padding='max_length')
        # 图像处理
        image = self.transform(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoding, image, label


class MultiModalModel(nn.Module):
    # 多模态模型
    def __init__(self, num_classes=30):
        super(MultiModalModel, self).__init__()
        # 文本分支
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.text_fc = nn.Linear(768, 512)
        # 图像分支
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # 去掉全连接层
        self.image_fc = nn.Linear(2048, 512)
        # 融合与分类
        self.fusion_fc = nn.Linear(512 + 512, 256)
        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, text_inputs, images):
        # 文本特征
        input_ids = text_inputs['input_ids'].squeeze(1)
        attention_mask = text_inputs['attention_mask'].squeeze(1)
        text_features = self.bert(input_ids, attention_mask=attention_mask)[
            1]  # [CLS] token
        text_features = self.text_fc(text_features)
        # 图像特征
        image_features = self.resnet(images)
        image_features = self.image_fc(image_features)
        # 特征融合
        fused_features = torch.cat((text_features, image_features), dim=1)
        fused_features = self.dropout(
            torch.relu(self.fusion_fc(fused_features)))
        logits = self.classifier(fused_features)
        return logits


# 数据预处理
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据（假设已有titles, images, labels）
dataset = ProductDataset(titles, images, labels, tokenizer, transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型、损失函数、优化器
model = MultiModalModel(num_classes=30).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(20):
    model.train()
    for text_inputs, images, labels in train_loader:
        text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(text_inputs, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 评估（省略）
