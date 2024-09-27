from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from torchinfo import summary

from model import SkipGramModel

model = SkipGramModel(184, 300)
model.load_state_dict(torch.load("./models/skip_gram.pth"))

# 将词向量进行 PCA 降维, 并进行二维可视化展示

# 设置全局字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载词汇表 word2idx
word2idx = {}
with open("models/word2idx.txt", "r") as f:
    for line in f:
        word, idx = line.strip().split()
        word2idx[word] = int(idx)

embeddings = model.in_embed.weight.data.cpu().numpy()
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)
plt.figure(figsize=(20, 20))
for i, word in enumerate(word2idx.keys()):
    x, y = pca_result[i]
    plt.scatter(x, y)
    plt.text(x, y, word)

plt.show()
