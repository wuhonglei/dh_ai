## 数据集

### 搜狐新闻数据(SogouCS) 版本：2012
链接: https://tianchi.aliyun.com/dataset/94521
描述: 来自搜狐新闻2012年6月—7月期间国内，国际，体育，社会，娱乐等18个频道的新闻数据，提供URL和正文信息。
压缩包大小: `699.33MB`
解压后的大小: `1.7G`
解压后的文件名: `sohu_data.json`

json 文件格式
```json
[
  {
    "title": "标题",
    "content": "内容"
  },
  {
    "title": "标题",
    "content": "内容"
  },
]
```

### 数据集处理
1. 为了方便处理，将 `sohu_data.json` 转换为 `sohu_data.csv`
2. 为了方便数据处理，将 `sohu_data.csv` 按照 0.7, 0.15, 0.15 的比例划分为 `train.csv`, `val.csv`, `test.csv`
3. `data/vocab.txt` 是基于 `sohu_data.csv` 生成的词表，词表中每个词后面存储了词频

### 词袋模型

### TF-IDF 词频-逆文档频率模型

### Word2Vec 词向量模型