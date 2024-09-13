import jieba # 导入jieba
text_zh = "我是一个学生。" #定义中文句子
# 调用lcut函数，对text_zh进行切词
zh_tokens = jieba.lcut(text_zh)
# 打印句子text_zh和切词结果zh_tokens
print(text_zh)
print(zh_tokens)

import spacy # 导入spacy库
# 调用spacy.load，加载英文词典，生成nlp对象
nlp = spacy.load("en_core_web_sm")
text_en = "I am a student."
# 将待处理文本text_en输入至nlp，进行分析
doc = nlp(text_en)
# 遍历doc中的token，获取分词结果
en_tokens = [token.text for token in doc]

print(text_en)
print(en_tokens) # 打印结果




















