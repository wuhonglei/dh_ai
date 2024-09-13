# 计算词语的相似度，对词语进行类比推理，并对词向量进行可视化
from gensim.models.keyedvectors import KeyedVectors #导入KeyedVectors

print("load word_vectors, please wait...")

# 使用load_word2vec_format函数，读取已经下载好的Google-news词向量字典
# 该接口会返回一个词向量模型对象model
model = KeyedVectors.load_word2vec_format(
            './word2vec-google-news-300.gz',
            binary = True)

# 使用模型的most_similar接口，获取与father最相似的前5个词语
sims = model.most_similar('father', topn = 5)

# 打印它们
print('father most similar words(top 5):')
for word, sim in sims:
    print('[%s] = %lf'%(word, sim))
print("")

# 使用similarity，判断两个词语的相似性
sim = model.similarity('father', 'mother')
print('father and mother sim = %lf\n'%(sim))


# 使用most_similar接口进行词语的类比推理
# 这里推理father-man=什么-woman
sims = model.most_similar(positive=['father', 'woman'],
                          negative=['man'], 
                          topn=5)

# 计算并打印最有可能的5个词语
print('father - man = ? - woman, the ? is (top 5)')
for word, sim in sims:
    print('[%s] = %lf'%(word, sim))
print("")



# 词向量的可视化
# 使用word保存8个词语
words = ['man', 'woman', 'king', 'queen',
        'cat', 'dog', 'mother', 'father']
        
vec_300d = list() # 保存词向量
for word in words: # 遍历这8个词语
    # 使用model中括号word，获取每个词的词向量
    vec_300d.append(model[word]) #保存到vec_300d

from sklearn.decomposition import PCA #导入PCA降维模块

pca = PCA(n_components = 2) #创建PCA模型

# 使用pca.fit_transform，将300维的词向量，降低到2维
vec_2d = pca.fit_transform(vec_300d)

# 将这两个维度保存到x和y的列表中
x = list()
y = list()
for vec in vec_2d:
    x.append(vec[0])
    y.append(vec[1])


# 使用matplotlib，将这些数据点绘制出来
import matplotlib.pyplot as plt
plt.scatter(x, y)

for i in range(len(x)):
    # 绘制使用plt.annotate接口
    plt.annotate(words[i], # 点的标签文本
                xy=(x[i], y[i]), # 点的坐标
                xytext=(-10, 10), # 标签文本的坐标
                textcoords='offset points')
plt.show()















