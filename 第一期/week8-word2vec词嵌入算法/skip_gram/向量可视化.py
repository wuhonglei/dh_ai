from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
import pickle

from model import SkipGramModel

checkpoint = torch.load("./models/skip_gram.pth",
                        map_location=torch.device('cpu'), weights_only=True)
vocab_size = checkpoint['vocab_size']
embedding_dim = checkpoint['embedding_dim']
model = SkipGramModel(vocab_size, embedding_dim)
model.load_state_dict(checkpoint['model_state_dict'])

# 将词向量进行 PCA 降维, 并进行二维可视化展示

# 设置全局字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载词汇表 word2idx
with open('./models/vocab.pkl', 'rb') as f:
    data = pickle.load(f)
    word2idx = data['word2idx']

embeddings = model.in_embed.weight.data.cpu().numpy()
pca = PCA(n_components=2)
pca_result = pca.fit_transform(embeddings)
plt.figure(figsize=(20, 10))
for i, word in enumerate(word2idx.keys()):
    x, y = pca_result[i]
    plt.scatter(x, y)
    plt.text(x, y, word)

plt.show()
