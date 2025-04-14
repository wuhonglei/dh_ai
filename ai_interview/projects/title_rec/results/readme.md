## 介绍
### 基线模型

![基线模型效果](https://p.ipic.vip/9otk4y.png)
结论:
svm 在使用 tf-idf 作为特征时，只需要使用 spacy 或 nltk 分词，效果最好, ，准确率最高可以达到 `0.8832`, 但是训练时间比较久，一轮训练的时间大概 `400s`
logistic 回归在使用 BOW 作为特征时，效果最好，准确率最高可以达到 `0.8708`, 训练比较快, 一轮训练时间大概 `18s`

### fasttext 模型

![fasttext 模型效果](https://p.ipic.vip/9otk4y.png)
结论:
1. fasttext 模型在使用 spacy 分词时并且移除停用词时，效果最好, ，准确率最高可以达到 `0.8755`，模型大小 `40MB`
2. fasttext 模型使用预训练向量 `crawl-300d-2M.vec` 时，准确率最高可达到 `0.8833`, 模型大小 `264MB`

### textcnn 模型

https://wandb.ai/wuhonglei1017368065-shopee/shopee_title_textcnn_model/table?nw=nwuserwuhonglei1017368065
最好的效果: `0.8416`
模型大小: `24MB`
训练时长: `3min(5epochs)`


### bert 模型

https://wandb.ai/wuhonglei1017368065-shopee/shopee_title_bert_model?nw=nwuserwuhonglei1017368065
对比了 `bert-base-uncased`, `distilbert-base-uncased`, `albert-xlarge-v2` 三个模型
`bert-base-uncased` 最好效果: `0.8848`, 模型大小 `417MB`, 训练时长 6min(3epochs)
`distilbert-base-uncased` 最好效果: `0.8817`, 模型大小 `253MB`, 训练时长 3min(3epochs)
`albert-xlarge-v2` 最好效果: `0.2102`, 模型大小 `224MB`, 训练时长 24min(3epochs)
综上: 
1. `bert-base-uncased` 效果最好，但是模型大小比较大，训练时间比较久
2. `distilbert-base-uncased` 效果其次，模型大小较小，训练时间较短

### 总结


### leaf_level 结果

1. 使用 fasttext 模型，使用联合搜索或者级联搜索，效果差不多，联合搜索最终只生成一个模型，级联搜索最终生成多个模型，准确率大概是 `0.75`, 级联训练使用 beam search k=2 时，能够提高 2% 的准确率
2. 使用 bert 模型
  - 一个模型集成 level1 和 leaf_level 预测，效果大概是 `0.724`
  - 联合搜索，效果大概是 `0.719`