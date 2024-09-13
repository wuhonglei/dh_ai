import jieba
# 实现中文分词函数tokenize_zh，函数传入文本text
def tokenize_zh(text):
    # 使用lcut对text分词，得到结果tokens
    tokens = jieba.lcut(text)
    res = []
    for token in tokens: # 遍历分词结果
        token = token.strip()
        if len(token) == 0: # 过滤掉空白字符
            continue
        res.append(token) # 将token添加到res中
    # # 将结果通过空格相连，保存到字符串中返回
    return ' '.join(res)

import spacy
# 实现英文分词函数tokenize_en，函数传入nlp对象和文本text
def tokenize_en(nlp, text):
    # 使用nlp进行分析，得到doc
    doc = nlp(text)
    res = []
    for token in doc: # 遍历doc中的token
        token = token.text.strip()
        if len(token) == 0: # 过滤掉空白字符
            continue
        res.append(token) # 将token添加到res中
    # 将结果通过空格相连，保存到字符串中返回
    return ' '.join(res)

from datasets import load_dataset
if __name__ == "__main__":
    # 调用load_dataset，导入数据
    dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")
    # 使用train_file和test_file，分别保存训练数据和测试数据
    train_file = open('train.data', 'w', encoding='utf-8')
    test_file = open('test.data', 'w', encoding='utf-8')
    # 调用spacy.load，加载英文模型
    nlp = spacy.load("en_core_web_sm")

    cnt = 0
    print("train data len: %d" % (len(dataset['train'])))
    for item in dataset['train']: # 遍历训练数据集
        # 获取到英文文本en_text和中文文本zh_text
        en_text = item['translation']['en']
        zh_text = item['translation']['zh']
        # 使用tokenize_en和tokenize_zh对文本分词
        en_token = tokenize_en(nlp, en_text)
        zh_token = tokenize_zh(zh_text)
        # 将英文和中文的分词结果，使用\t符号分隔，保存到文件中
        train_file.write(en_token + '\t' + zh_token + '\n')
        cnt += 1
        # 每写入100个数据，打印一次调试信息
        if cnt % 100 == 0:
            print("deal %d train data."%(cnt))

    # 按照同样的方法处理测试数据集，并将结果保存到文件中
    cnt = 0
    print("test data len: %d" % (len(dataset['test'])))
    for item in dataset['test']:
        en_text = item['translation']['en']
        zh_text = item['translation']['zh']
        en_token = tokenize_en(nlp, en_text)
        zh_token = tokenize_zh(zh_text)
        test_file.write(en_token + '\t' + zh_token + '\n')

        cnt += 1
        if cnt % 100 == 0:
            print("deal %d test data." % (cnt))

    train_file.close()
    test_file.close()

