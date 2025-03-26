## 介绍
使用 bow 模型，将新闻的内容转换为向量，然后使用milvus进行存储，并进行相似度搜索。

## 一、TF 性能评估
使用 [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) 性能对比, MRR 的范围是 0-1, 值越大越好。

![MRR](./screenshots/mrr.png)

| 版本 | MRR | 词汇表 | 向量维度 | min_freq | 停用词 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| milvus.bow.v1 | 0.0926 | `sohu_data.csv` | 21968 | 1000 | 否 | - |
| milvus.bow.v2 | 0.1674 | `sohu_data.csv` | 21243 | 1000 | 是 | - |
| milvus.bow.v3 | 0.0893 | `val.csv` | 4439 | 1000 | 是 | - |
| milvus.bow.v3_1 | 0.1322 | `val.csv` | 10634 | 350 | 是 | - |
| milvus.bow.v3_2 | 0.1560 | `val.csv` | 16439 | 200 | 是 | - |
| milvus.bow.v3_3 | 0.1560 | `val.csv` | 16439 | 200 | 是 | tf=word_count/total_words_in_curr_doc |


## 二、TF-IDF 性能评估
使用 [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) 性能对比, MRR 的范围是 0-1, 值越大越好。

![MRR](./screenshots/mrr_tfidf.png)

| 版本 | MRR | 词汇表 | 向量维度 | min_freq | 停用词 | 备注 |
| --- | --- | ---   | --- | --- | --- | --- |
| milvus.bow.v4 | 0.2219 | `val.csv` | 16439 | 200 | 是 | - |


## 三、word2vector 性能评估
使用 [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) 性能对比, MRR 的范围是 0-1, 值越大越好。

![MRR](./screenshots/mrr_word2vector.png)

| 版本 | MRR | loss | 词汇表 | 向量维度 | 窗口大小 | epoch | min_freq | 停用词 | 备注 |
| --- | --- | ---  | --- | --- | --- | --- | --- | --- | --- | --- |
| milvus.bow.v5 | 0.0676 | 6 | `val.csv` | 100 | 2 | 4 | 200 | 是 | - |
| milvus.bow.v5_1 | 0.0603 | 5.19 | `val.csv` | 100 | 2 | 4 | 200 | 是 | - |
| milvus.bow.v5_2 | 0.1364 | 4.2 | `val.csv` | 100 | 2 | 4 | 200 | 是 | adamW 优化器 |
| milvus.bow.v5_3 | 0.1990 | 3.7 | `val.csv` | 200 | 5 | 10 | 350 | 是 | adamW 优化器 |

## 四、CBOW-Siamese 性能评估
使用 [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) 性能对比, MRR 的范围是 0-1, 值越大越好。

![MRR](./screenshots/mrr_cbow_siamese.png)


| 版本 | MRR | loss | 训练集 | 验证集 | 评估集 | 向量维度 | epoch | min_freq | 停用词 | 备注 |
| --- | --- | ---  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| milvus.bow.v7 | 0.2826 | train_loss: 13.49 | `val_10000.csv` | `test_1000.csv` | `val_10000.csv` | 200 | 10 | 350 | 是 | 词向量平均, 不使用预训练 embedding |
| milvus.bow.v7_1 | 0.2413 | train_loss: 7.75 | `val_10000.csv` | `test_1000.csv` | `val_10000.csv` | 200 | 10 | 350 | 是 | 词向量平均, 使用预训练 embedding |
| milvus.bow.v7_2 | 0.0963 | train_loss: 7.75 | `val_10000.csv` | `test_1000.csv` | `val.csv` | 200 | 10 | 350 | 是 | 词向量平均, 使用预训练 embedding|
| milvus.bow.v7_3 | 0.1460 | train_loss: 11.58 | `val_10000.csv` | `test_1000.csv` | `val.csv` | 200 | 10 | 350 | 是 | 词向量平均, 不使用预训练 embedding|
| milvus.bow.v7_4 | 0.2809 | train_loss: 8.222 | `val_10000.csv` | `test_1000.csv` | `val_10000.csv` | 200 | 10 | 350 | 是 | 词向量平均, 不使用预训练 embedding, temperature = 0.1|
| milvus.bow.v7_5 | 0.2809 | train_loss: 4.3165 | `val_10000.csv` | `test_1000.csv` | `val_10000.csv` | 200 | 10 | 350 | 是 | 词向量平均, 不使用预训练 embedding, temperature = 1|
| milvus.bow.v7_6 | 0.0489 | train_loss: 5.2095 | `val_10000.csv` | `test_1000.csv` | `val_10000.csv` | 200 | 10 | 350 | 是 | 词向量最大池化, 不使用预训练 embedding, temperature = 1|