## 介绍
使用 bow 模型，将新闻的内容转换为向量，然后使用milvus进行存储，并进行相似度搜索。

## 一、bert 性能评估
使用 [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) 性能对比, MRR 的范围是 0-1, 值越大越好。

![MRR](./screenshots/mrr.png)

| 版本 | MRR | 词汇表 | 向量维度 | min_freq | 停用词 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| milvus.bow.v9 | 0.0059 | `val.csv` | 768 | - | 否 | 直接使用 bert 模型 last_hidden_state 的 [CLS] 位置的输出 |
| milvus.bow.v9_1 | 0.2805 | `val.csv` | 768 | - | 否 | 使用 test_1000.csv 训练, 使用 val_1000.csv 评估 |
