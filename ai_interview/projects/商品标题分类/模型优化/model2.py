import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# 定义类别层级映射关系
level1_to_level2 = {
    0: [0, 1],  # 一级类别 0 对应二级类别 0 和 1
    1: [2, 3],  # 一级类别 1 对应二级类别 2 和 3
    2: [4]      # 一级类别 2 对应二级类别 4
}
level2_to_level3 = {
    0: [0, 1],  # 二级类别 0 对应三级类别 0 和 1
    1: [2],     # 二级类别 1 对应三级类别 2
    2: [3, 4],  # 二级类别 2 对应三级类别 3 和 4
    3: [5],     # 二级类别 3 对应三级类别 5
    4: [6, 7]   # 二级类别 4 对应三级类别 6 和 7
}

# 定义层级约束损失函数


def hierarchical_loss(output_level1, output_level2, output_level3, labels, device):
    """
    计算层级约束损失。
    :param output_level1: 一级类别的 logits 输出
    :param output_level2: 二级类别的 logits 输出
    :param output_level3: 三级类别的 logits 输出
    :param labels: 真实标签，形状为 [batch_size, 3]，包含 [一级类别, 二级类别, 三级类别]
    :param device: 设备
    :return: 总损失
    """
    batch_size = labels.size(0)
    loss = 0.0

    # 交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 一级类别损失
    loss_level1 = criterion(output_level1, labels[:, 0])

    # 二级类别损失
    loss_level2 = criterion(output_level2, labels[:, 1])

    # 三级类别损失
    loss_level3 = criterion(output_level3, labels[:, 2])

    # 层级约束损失
    constraint_loss = 0.0
    for i in range(batch_size):
        # 获取真实标签
        level1_label = labels[i, 0].item()
        level2_label = labels[i, 1].item()
        level3_label = labels[i, 2].item()

        # 获取预测结果
        level1_pred = torch.argmax(output_level1[i]).item()
        level2_pred = torch.argmax(output_level2[i]).item()
        level3_pred = torch.argmax(output_level3[i]).item()

        # 检查二级类别是否属于一级类别的子类别
        if level2_pred not in level1_to_level2.get(level1_pred, []):
            constraint_loss += 1.0  # 增加惩罚

        # 检查三级类别是否属于二级类别的子类别
        if level3_pred not in level2_to_level3.get(level2_pred, []):
            constraint_loss += 1.0  # 增加惩罚

    # 将总损失归一化
    constraint_loss = constraint_loss / batch_size

    # 总损失 = 分类损失 + 层级约束损失
    total_loss = loss_level1 + loss_level2 + loss_level3 + constraint_loss

    return total_loss

# 示例模型定义（与前述一致）


class HierarchicalMultimodalClassifier(nn.Module):
    def __init__(self, num_classes_level1, num_classes_level2, num_classes_level3):
        super(HierarchicalMultimodalClassifier, self).__init__()
        self.image_model = models.resnet50(pretrained=True)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 256)

        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 256)

        self.fc_level1 = nn.Linear(256 + 256, num_classes_level1)
        self.fc_level2 = nn.Linear(
            256 + 256 + num_classes_level1, num_classes_level2)
        self.fc_level3 = nn.Linear(
            256 + 256 + num_classes_level2, num_classes_level3)

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_model(image)
        text_outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask)
        text_features = self.text_fc(text_outputs.pooler_output)

        combined_features = torch.cat((image_features, text_features), dim=1)

        output_level1 = self.fc_level1(combined_features)

        level1_softmax = torch.softmax(output_level1, dim=1)
        combined_features_level2 = torch.cat(
            (combined_features, level1_softmax), dim=1)
        output_level2 = self.fc_level2(combined_features_level2)

        level2_softmax = torch.softmax(output_level2, dim=1)
        combined_features_level3 = torch.cat(
            (combined_features, level2_softmax), dim=1)
        output_level3 = self.fc_level3(combined_features_level3)

        return output_level1, output_level2, output_level3


# 示例训练代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HierarchicalMultimodalClassifier(
    num_classes_level1=3, num_classes_level2=5, num_classes_level3=8).to(device)

# 示例数据
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
titles = ["This is a smartphone", "This is a laptop"]
labels = [[0, 1, 2], [0, 3, 4]]  # 每个样本的 [一级类别, 二级类别, 三级类别]
dataset = MultimodalDataset(image_paths, titles, labels, transform, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):  # 训练 5 个 epoch
    model.train()
    for images, input_ids, attention_mask, labels in dataloader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # 前向传播
        output_level1, output_level2, output_level3 = model(
            images, input_ids, attention_mask)

        # 计算损失（包含层级约束）
        loss = hierarchical_loss(
            output_level1, output_level2, output_level3, labels, device)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
