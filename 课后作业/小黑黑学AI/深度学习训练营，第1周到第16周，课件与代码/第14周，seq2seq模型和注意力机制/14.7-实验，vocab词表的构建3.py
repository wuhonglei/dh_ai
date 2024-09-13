import pickle
# 直接通过pickle.load，读取词表v
with open("v.pkl", "rb") as f:
    v = pickle.load(f)

s = ["deep", "learning", "with", "pytorch", "is", "awesome"]

print("change s to index:")
index = list()
# 从文件读取的词表v，使用方式是一样的，同样可以将句子s转为索引形式
for word in s:
    index.append(v[word])
print(s)
print(index)



















