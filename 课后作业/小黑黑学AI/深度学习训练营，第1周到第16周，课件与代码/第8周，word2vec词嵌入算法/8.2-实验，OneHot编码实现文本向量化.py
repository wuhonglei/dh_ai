import jieba #导入jieba模块

# 函数传入待提取特征词的文档doc和表示文档库全部单词的集合wordbag
# 函数会将doc中的关键词提取出来，并将关键词添加到wordbag中
def doc_to_words(doc, wordbag):
    stopwords = {'的', '很', '是', '个', '好', '。'}
    words = set() #设置集合words保存提取结果
    tokens = jieba.lcut(doc) #将doc进行切词
    for token in tokens: #遍历切词结果
        if token in stopwords: #过滤掉停用
            continue
        #将过滤后的关键词添加至words和wordbag
        words.add(token)
        wordbag.add(token)
    return words #返回words


#将提取出的单词words，根据文档库词表wordbag，转换为特征向量的形式
def words_to_feature(words, wordbag):
    feature = list() #保存特征向量
    for w in wordbag: #遍历文档库中的词语
        if w in words: #如果单词出现在words中
            feature.append(1) #向feature中添加1
        else: #否则添加0
            feature.append(0)
    return feature #返回特征向量feature

if __name__ == '__main__':
    # 设置3个文档，作为文档库
    doc1 = '北京的后海是个好地方。'
    doc2 = '北京的小吃很好吃。'
    doc3 = '北京大学是个好学校。'
    wordbag = set() #设置保存文档库中所有单词的集合
    #调用doc_to_words将文档转换特征词列表
    words1 = doc_to_words(doc1, wordbag)
    words2 = doc_to_words(doc2, wordbag)
    words3 = doc_to_words(doc3, wordbag)
    #将特征词列表转换为特征向量
    feature1 = words_to_feature(words1, wordbag)
    feature2 = words_to_feature(words2, wordbag)
    feature3 = words_to_feature(words3, wordbag)
    #打印全部的结果
    print("wordbag: %s" % (wordbag))
    print("doc1 = %s" % (doc1))
    print("doc2 = %s" % (doc2))
    print("doc3 = %s" % (doc3))
    print("words1 = %s" % (words1))
    print("words2 = %s" % (words2))
    print("words3 = %s" % (words3))
    print("feature1 = %s" % (feature1))
    print("feature2 = %s" % (feature2))
    print("feature3 = %s" % (feature3))






