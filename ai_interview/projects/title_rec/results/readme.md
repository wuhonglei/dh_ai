## 介绍
### 基线模型

![基线模型效果](https://p.ipic.vip/9otk4y.png)
结论:
svm 在使用 tf-idf 作为特征时，只需要使用 spacy 或 nltk 分词，效果最好, ，准确率最高可以达到 `0.8832`, 但是训练时间比较久，一轮训练的时间大概 `400s`
logistic 回归在使用 BOW 作为特征时，效果最好，准确率最高可以达到 `0.8708`, 训练比较快, 一轮训练时间大概 `18s`

### fasttext 模型

![fasttext 模型效果](https://p.ipic.vip/9otk4y.png)
结论:
fasttext 模型在使用 spacy 分词时并且移除停用词时，效果最好, ，准确率最高可以达到 `0.8755`
