# 函数make_train_data传入raw_text
# 函数计算raw_text中包含的词语集合vocab
# 词语数量vocab_size
# 词语到索引的字典word2ix
# 索引到词语的字典ix2word
def stat_raw_text(raw_text):
    # 将raw_text中保存的词语，放到集合set中去重
    vocab = set(raw_text) # 得到了词语的集合vocab
    vocab_size = len(vocab) # 计算词语集合的长度vocab_size
    word2ix = dict() #设置词语到索引的字典
    for ix, word in enumerate(vocab): #遍历词表vocab构造字典
        word2ix[word] = ix
    ix2word = dict() #设置索引到词语的字典
    for ix, word in enumerate(vocab): #遍历词表vocab构造字典
        ix2word[ix] = word
    return vocab, vocab_size, word2ix, ix2word


if __name__ == '__main__':
    # 经过切词后，得到raw_text，其中包括了5个词语
    # raw_text =  ['a', 'b', 'c', 'b', 'a']
    raw_text = """a b c b a""".split()
    print("raw_text = ", raw_text)
    vocab, vocab_size, word2ix, ix2word = stat_raw_text(raw_text)
    # 集合vocab，包括a、b、c三个词
    # vocab =  {'a', 'b', 'c'}
    print("vocab = ", vocab)
    # 长度vocab_size =  3
    print("vocab_size = ", vocab_size)
    # word2ix =  {'a': 0, 'b': 1, 'c': 2}
    print("word2ix = ", word2ix)
    # ix2word =  {0: 'a', 1: 'b', 2: 'c'}
    print("ix2word = ", ix2word)

