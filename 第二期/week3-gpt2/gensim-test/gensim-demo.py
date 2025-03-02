import gensim.downloader
import gensim.models


# 之后使用
model = gensim.models.KeyedVectors.load_word2vec_format(
    'model.bin', binary=True)

"""
我想要寻找一个词， ?  = queen - woman + man
"""
result = model.most_similar(
    positive=['queen', 'man'], negative=['woman'], topn=1)
print(f"queen - woman + man = {result[0][0]} (相似度: {result[0][1]:.4f})")
