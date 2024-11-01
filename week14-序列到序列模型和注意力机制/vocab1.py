import pickle
from collections import Counter

# 基于一个例子，来说明vocab词表的使用方法
# texts中包含了3个句子，我们要基于texts构建词表vocab
texts = [
    ["this", "is", "a", "test"],
    ["pytorch", "is", "fun"],
    ["pytorch", "is", "great", "for", "deep", "learning"]
]

# 使用vocab将另一个句子s中的词转换为索引
s = ["deep", "learning", "with", "pytorch", "is", "awesome"]

word_list = []
for text in texts:  # 遍历texts中的每个句子
    word_list.extend(text)

counter = Counter(word_list)  # 创建一个counter对象
vocab = {
    '<unk>': 0
}
for i, word in enumerate(counter):
    vocab[word] = len(vocab)

print('vocab', vocab)
s_index = [vocab.get(word, vocab["<unk>"]) for word in s]
print(s_index)

with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
