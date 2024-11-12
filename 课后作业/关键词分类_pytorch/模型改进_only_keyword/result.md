### 结果说明
#### 一、字符 LSTM 分类结果
1. 仅仅使用 SEO 搜索关键词进行训练的结果如下（模型结构 1）:
30/30 [52:20<00:00, 104.67s/it, test_acc=71.18%, train_acc=96.51%]

```python
KeywordCategoryModel(
  (embedding): Embedding(50, 25, padding_idx=0)
  (dropout1): Dropout(p=0.15, inplace=False)
  (lstm): LSTM(25, 128, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
  (dropout2): Dropout(p=0.35, inplace=False)
  (fc): Linear(in_features=256, out_features=26, bias=True)
)
```

```json
{
    "vocab_size": 50,
    "embed_dim": 25,
    "hidden_size": 128,
    "num_classes": 26,
    "padding_idx": 0,
    "num_epochs": 30,
    "learning_rate": 0.01,
    "batch_size": 512
}
```

2. 仅仅使用 SEO 搜索关键词进行训练的结果如下（模型结构 2）:
30/30 [30:19<00:00, 60.64s/it, test_acc=73.18%, train_acc=96.48%]
对应的模型存储名称: `SG_LSTM_128*2_model.pth`

```python
KeywordCategoryModel(
  (embedding): Embedding(50, 25, padding_idx=0)
  (dropout1): Dropout(p=0.15, inplace=False)
  (lstm): LSTM(25, 128, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
  (dropout2): Dropout(p=0.35, inplace=False)
  (fc1): Linear(in_features=256, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=26, bias=True)
)
```

```json
{
    "vocab_size": 50,
    "embed_dim": 25,
    "hidden_size": 128,
    "num_classes": 26,
    "padding_idx": 0,
    "num_epochs": 30,
    "learning_rate": 0.01,
    "batch_size": 512
}
```