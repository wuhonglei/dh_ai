from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from gensim.models import Word2Vec


from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from dataset import NewsDataset, build_vocab, collate_batch
from model import TextClassifier, build_word2vec_embeddings

# 1. 加载数据集
newsgroups = fetch_20newsgroups(
    subset='all', remove=('headers', 'footers', 'quotes'), data_home='/mnt/model/nlp/scikit_learn_data/')

texts = newsgroups.data   # type: ignore
labels = newsgroups.target  # type: ignore

# 3. 数据分割
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 创建数据集
train_dataset = NewsDataset(train_texts, train_labels, use_cache=True)
test_dataset = NewsDataset(test_texts, test_labels, use_cache=True)


# 创建数据加载器
special_tokens = ['<pad>', '<unk>']
vocab = build_vocab(train_dataset, special_tokens)

# 打印两个词表长度
print("vocab:", len(vocab))
# 词表的保存
# with open("vocab.pkl", "wb") as f:
#     pickle.dump(vocab, f)


def collate(batch):
    """ 回调函数，用于不同长度的文本进行填充 """
    return collate_batch(batch, vocab)


batch_size = 64
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


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
embed_dim = 100
num_classes = np.max(labels) + 1  # 类别数, labels 从 0 开始
num_epochs = 300
learning_rate = 0.01
vocab_size = len(vocab)
padding_idx = vocab.token2id["<pad>"]
w2v_model = Word2Vec.load('./models/word2vec.model')
word2vec_embeddings = build_word2vec_embeddings(vocab.token2id, w2v_model.wv)
# model = LinearClassifier(input_size, num_classes).to(device)
model = TextClassifier(vocab_size, embed_dim,
                       num_classes, padding_idx).to(device)
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

# 保存模型
torch.save(model.state_dict(),
           f'./models/{num_epochs}_model.pth')
