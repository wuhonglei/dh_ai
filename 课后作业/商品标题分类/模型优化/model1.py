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

# 模型定义（与前述一致）


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

# 推理阶段逐级过滤逻辑


def hierarchical_prediction(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, input_ids, attention_mask, _ in dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 逐级预测
            output_level1, output_level2, output_level3 = model(
                images, input_ids, attention_mask)

            # 一级类别预测
            predictions_level1 = torch.argmax(
                output_level1, dim=1).cpu().numpy()

            # 二级类别预测（基于一级类别过滤）
            predictions_level2 = []
            for i, level1_pred in enumerate(predictions_level1):
                # 获取当前样本可能的二级类别
                valid_level2_classes = level1_to_level2[level1_pred]
                # 只保留可能的二级类别的 logits
                level2_logits = output_level2[i, valid_level2_classes]
                level2_pred = valid_level2_classes[torch.argmax(
                    level2_logits).item()]
                predictions_level2.append(level2_pred)

            # 三级类别预测（基于二级类别过滤）
            predictions_level3 = []
            for i, level2_pred in enumerate(predictions_level2):
                # 获取当前样本可能的三级类别
                valid_level3_classes = level2_to_level3[level2_pred]
                # 只保留可能的三级类别的 logits
                level3_logits = output_level3[i, valid_level3_classes]
                level3_pred = valid_level3_classes[torch.argmax(
                    level3_logits).item()]
                predictions_level3.append(level3_pred)

            # 保存结果
            for i in range(len(predictions_level1)):
                predictions.append({
                    "level1": predictions_level1[i],
                    "level2": predictions_level2[i],
                    "level3": predictions_level3[i]
                })

    return predictions


# 示例推理
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
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

predictions = hierarchical_prediction(model, dataloader, device)
for pred in predictions:
    print(pred)
