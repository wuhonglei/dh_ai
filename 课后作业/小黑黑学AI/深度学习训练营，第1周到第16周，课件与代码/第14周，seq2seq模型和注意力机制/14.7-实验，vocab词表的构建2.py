# 定义保存3个文本序列的texts
texts = [
    ["this", "is", "a", "test"],
    ["pytorch", "is", "fun"],
    ["pytorch", "is", "great", "for", "deep", "learning"]
]

from torchtext.vocab import build_vocab_from_iterator
# 使用build_vocab_from_iterator，创建texts对应的词表v
v = build_vocab_from_iterator(texts, specials=['<unk>'])
v.set_default_index(v["<unk>"]) # 设置默认单词unk

s = ["deep", "learning", "with", "pytorch", "is", "awesome"]
print("change s to index:")
index = list()
# 使用词表v，将句子s转换为索引序列index
for word in s:
    index.append(v[word])
print(s)
print(index)

import pickle
# 使用pickle.dump，将词表v用文件的形式保存
with open("v.pkl", "wb") as f:
    pickle.dump(v, f)

