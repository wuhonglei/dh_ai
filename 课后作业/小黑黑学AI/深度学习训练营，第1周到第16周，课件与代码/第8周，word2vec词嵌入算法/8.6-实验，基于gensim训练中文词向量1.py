import os #导入os，用于打开文件夹
import codecs #导入codecs，用于读取文件
import jieba #导入jieba，用于中文分词

#函数传入待读取的文件filename
#该函数会将传入文件中的文本，按行切词，保存到sents中
def read_file_and_cut(filename, sents):
    # 使用codecs.open打开文件，并读取文件中的每一行，保存到lines中
    lines = codecs.open(filename, 'r', encoding='utf-8').readlines()
    
    for line in lines: # 遍历lines
        line = line.strip() # 将每行数据去掉头尾空白字符
        # 使用jieba.lcut切词，切词结果保存到words列表中
        words = jieba.lcut(line) 
        sent = list() # 设置sent保存每一行的切词结果
        for word in words: # 遍历words
            word = word.strip() 
            if len(word) == 0:
                continue
            # 去掉空白词语后，将词语添加到sent中
            sent.append(word)
        sents.append(sent)
        
# 设置函数load_files_to_sents
# 读取dir_path路径下的全部文件，并对全部文件中的全部词语切词
def load_files_to_sents(dir_path):
    # 使用os.listdir，获得dir_path中的的全部文件
    files = os.listdir(dir_path)
    sents = list() # 设置保存切词结果的列表
    for file in files: #遍历files列表
        # 生成每个文件的路径后
        file_path = os.path.join(dir_path, file)
        # 使用read_file_and_cut，对文件切词，结果保存到sents中
        read_file_and_cut(file_path, sents)
    return sents #返回sents


if __name__ == '__main__':
    # 通过load_files_to_sents
    # 将当前目录下的data文件夹中的全部文件，进行切词
    sents = load_files_to_sents('./data/')
    # 接着将切词结果打印出来
    for sent in sents:
        print(sent)
        

