# 基于一个例子，来说明vocab词表的使用方法
# texts中包含了3个句子，我们要基于texts构建词表vocab
texts = [
    ["this", "is", "a", "test"],
    ["pytorch", "is", "fun"],
    ["pytorch", "is", "great", "for", "deep", "learning"]
]
# 使用vocab将另一个句子s中的词转换为索引
s = ["deep", "learning", "with", "pytorch", "is", "awesome"]

# 为了构建词表，需要使用collections中的Counter
# 统计texts中词语出现的数量
from collections import Counter

# Counter的用法类似于字典，可以对任意对象
# 如列表、元组、字符串等等，进行统计，计算出每个元素的出现次数
counter = Counter() # 创建一个counter对象
for text in texts: # 遍历texts中的每个句子
    counter.update(text) # 使用counter进行统计
print(counter) # 打印counter

# 从torchtext模块中导入vocab
from torchtext.vocab import vocab

# 将counter传入vocab，创建词汇表v
# 这里额外传入参数specials，代表特殊标记词语
v = vocab(counter, specials=['<unk>'])
# 打印len(v)，可以得到词表中包含的词语数量
print("words number:", len(v))

# 使用<unk>表示未知词，并将<unk>对应的索引设置为默认索引
# 当遇到词表中没有出现的词语，就会返回<unk>
v.set_default_index(v['<unk>'])

# 通过词表v[]，可以访问<unk>、pytorch和abc，三个词语的索引
print(f"<unk> -> {v['<unk>']}")
print(f"hello -> {v['pytorch']}")
# abc没有在词表中出现，因此返回了unk对应的默认索引0
print(f"abc -> {v['abc']}")

# 通过get_itos获得词表中保存单词的列表itos
itos = v.get_itos()
# 打印itos的类型type(itos)，结果是列表list
print(f"itos type: {type(itos)}")
# 打印itos，可以看到词表v中包含的索引单词
print(itos)

# 通过itos，可以获得索引index到单词word的映射
print("index->word:")
for i in range(len(itos)):
    print(f"{i} -> {itos[i]}")
print("")

# 使用get_stoi获取单词到索引的映射stoi
stoi = v.get_stoi()
# 打印stoi的类型type(stoi)，结果是字典dict
print(f"stoi type: {type(stoi)}")
# 打印stoi，可以看到词表v中包含的全部单词到索引的字典
print(stoi)
print("word->index:")
for word in stoi: # 遍历stoi中的全部单词word
    # 打印单词word，并通过stoi[word]，获取word对应的索引
    print(f"{word} -> {stoi[word]}")
print("")

# 将第四个句子转换为索引序列
print("change s to index:")
index = list() # 定义列表index
for word in s: # 遍历句子s中的单词word
    # 将word对应的索引v[word]，添加到index中
    index.append(v[word])
# 打印s和index
print(s)
print(index)

