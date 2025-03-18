## 向量数据库 collection 的创建

### BOW 词袋模型

```python
# 创建集合
from pymilvus import MilvusClient
client = MilvusClient("milvus_demo.db")
client.create_collection(
  collection_name="bow_dimension_5_collection",
  dimension=5
)
```

### TF-IDF 词频-逆文档频率模型

### Word2Vec 词向量模型

