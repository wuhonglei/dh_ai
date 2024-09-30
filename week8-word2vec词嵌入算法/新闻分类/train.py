from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from dataset import NewsDataset
from model import LinearClassifier, SimpleNN

# 1. 加载数据集
newsgroups = fetch_20newsgroups(
    subset='all', remove=('headers', 'footers', 'quotes'), data_home='/mnt/model/nlp/scikit_learn_data/')

texts = newsgroups.data
labels = newsgroups.target

# 2. 标签编码（如果需要）
# 这里标签已经是整数编码，无需额外编码
num_classes = np.max(labels) + 1

# 3. 数据分割
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 4.1 词袋模型
bow_vectorizer = CountVectorizer(max_features=5000)  # 限制词汇表大小
X_train_bow = bow_vectorizer.fit_transform(train_texts)
X_test_bow = bow_vectorizer.transform(test_texts)

# 4.2 TF-IDF模型
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
X_test_tfidf = tfidf_vectorizer.transform(test_texts)

# 创建数据集
train_dataset_bow = NewsDataset(X_train_bow, train_labels)
test_dataset_bow = NewsDataset(X_test_bow, test_labels)

train_dataset_tfidf = NewsDataset(X_train_tfidf, train_labels)
test_dataset_tfidf = NewsDataset(X_test_tfidf, test_labels)

# 创建数据加载器
batch_size = 64
train_loader_bow = DataLoader(
    train_dataset_bow, batch_size=batch_size, shuffle=True)
test_loader_bow = DataLoader(
    test_dataset_bow, batch_size=batch_size, shuffle=False)

train_loader_tfidf = DataLoader(
    train_dataset_tfidf, batch_size=batch_size, shuffle=True)
test_loader_tfidf = DataLoader(
    test_dataset_tfidf, batch_size=batch_size, shuffle=False)


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            total_correct += torch.eq(pred, y).sum().float().item()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy


def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.numpy())
    return all_labels, all_preds


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size = 128
num_classes = num_classes
num_epochs = 300
learning_rate = 0.001

# 选择特征类型：'bow' 或 'tfidf'
feature_type = 'tfidf'  # Change to 'bow' for BOW

if feature_type == 'bow':
    train_loader = train_loader_bow
    test_loader = test_loader_bow
    input_size = X_train_bow.shape[1]
else:
    train_loader = train_loader_tfidf
    test_loader = test_loader_tfidf
    input_size = X_train_tfidf.shape[1]

print('input_size', input_size)
# model = LinearClassifier(input_size, num_classes).to(device)
model = SimpleNN(input_size, hidden_size, num_classes).to(device)
# model.load_state_dict(torch.load('./models/tfidf_50_model.pth',
#                       map_location=device, weights_only=True))
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练和评估
print('starting training...')
for epcoh in range(num_epochs):
    train_loss = train(model, train_loader, criteria, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criteria, device)
    print(f'Epoch {epcoh + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

# 获取所有预测和真实标签
true_labels, preds = get_predictions(model, test_loader, device)
# 计算准确率
accuracy = accuracy_score(true_labels, preds)
print(f'Accuracy: {accuracy*100:.2f}%')

# # 计算精确率、召回率和F1分数
# precision, recall, f1, _ = precision_recall_fscore_support(
#     true_labels, preds, average='weighted')
# print(f'Precision: {precision*100:.2f}%')
# print(f'Recall: {recall*100:.2f}%')
# print(f'F1 Score: {f1*100:.2f}%')


# 保存模型
torch.save(model.state_dict(),
           f'./models/{feature_type}_{num_epochs}_model.pth')
