# 使用词向量前，需要下载词向量的字典
# 使用gensim提供的接口进行下载
import gensim.downloader as api #导入gensim.downloader

# 使用api.load下载词典，并将下载好的词典路径打印出来
print(api.load("glove-twitter-25", return_path=True))











