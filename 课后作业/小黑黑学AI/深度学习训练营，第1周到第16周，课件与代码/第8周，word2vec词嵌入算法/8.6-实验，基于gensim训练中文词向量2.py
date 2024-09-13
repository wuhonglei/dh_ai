import os
import codecs
import jieba

#传入待读取的文件路径，将传入的文件中的文本按行切词，保存到sents中
def read_file_and_cut(filename, sents):
    lines = codecs.open(filename, 'r', encoding='utf-8').readlines()
    for line in lines: #读取文件中的每一行
        line = line.strip() #将每行中的头尾空字符去掉
        words = jieba.lcut(line)
        sent = list()
        for word in words:
            word = word.strip()
            if len(word) == 0:
                continue
            sent.append(word)
        sents.append(sent)

# 读dir_path路径下的全部文件，并对全部文件中的全部词语切词
def load_files_to_sents(dir_path):
    files = os.listdir(dir_path)
    sents = list()
    for file in files:
        # 生成文件路径后
        file_path = os.path.join(dir_path, file)
        read_file_and_cut(file_path, sents)
    return sents
    
# 完成数据处理的函数编写后，进行word2vec模型的训练
# 从gensim.models中导入word2vec模块
from gensim.models import word2vec
if __name__ == '__main__':
    sents = load_files_to_sents('./data/')
    # 完成切词后，将切词结果sents，传入Word2Vec的接口进行训练
    model = word2vec.Word2Vec(sents,
                              min_count = 2, #词语至少出现过2次
                              vector_size = 5) #训练的词向量维度是5

    word_vec = model.wv #通过model.wv获取训练好的词向量
    # 遍历这些词向量
    for index, word in enumerate(word_vec.index_to_key):
        print("%d %s %s"%(index, word, str(word_vec[word])))
    model.save("word2vec.model") #将模型保存为文件形式


