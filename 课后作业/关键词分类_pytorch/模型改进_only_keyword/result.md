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

3. 仅仅使用 SEO 搜索关键词进行训练的结果如下（模型结构 3）:
epcoh: 25; test acc: 74.64%; train acc: 97.65%

```python
KeywordCategoryModel(
  (embedding): Embedding(50, 25, padding_idx=0)
  (dropout1): Dropout(p=0.15, inplace=False)
  (lstm): LSTM(25, 128, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
  (dropout2): Dropout(p=0.35, inplace=False)
  (fc1): Linear(in_features=768, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=26, bias=True)
)

ouput, (hidden, _) = self.lstm(x)
hidden = self.dropout2(hidden)
last_layer_hidden = torch.cat(
    (hidden[-2], hidden[-1]), dim=-1)  # [batch, hidden_size * 2]
avg_seq_output = torch.mean(ouput, dim=1)  # [batch, hidden_size * 2]
max_seq_output, _ = torch.max(ouput, dim=1)  # [batch, hidden_size * 2]
concat_hidden = torch.cat(
    (last_layer_hidden, avg_seq_output, max_seq_output), dim=-1)  # [batch, hidden_size * 6]
output = self.fc1(concat_hidden)
output = self.fc2(output)
return output
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
    "batch_size": 1024
}
```

4. 使用 shopee clean title 进行训练的结果如下（模型结构 3）:

```txt
2024-11-13 09:06:19; epcoh: 1; test acc: 84.91%; train acc: 85.07%
2024-11-13 10:24:10; epcoh: 2; test acc: 86.83%; train acc: 87.24%
2024-11-13 11:40:19; epcoh: 3; test acc: 87.46%; train acc: 87.98%
```

对应的模型存储名称: `SG_LSTM_128*2_fc_2_shopee_title_model.pth`
词汇表名称: `SG_vocab_866321_f76939827d4d80a2b54308058027278b.json`

```json
{
    "vocab_size": 116,
    "embed_dim": 25,
    "hidden_size": 128,
    "num_classes": 26,
    "padding_idx": 0,
    "num_epochs": 3,
    "learning_rate": 0.01,
    "batch_size": 2048
}
```

```python
class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, output_size: int, padding_idx: int):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout1 = nn.Dropout(0.15)
        self.lstm = nn.LSTM(embed_dim, hidden_size,
                            batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)
        self.dropout2 = nn.Dropout(0.35)
        self.fc1 = nn.Linear(hidden_size * 6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout1(x)
        """
        hidden: [num_layers * num_directions, batch, hidden_size]

        对于 2 layers 和 bidirectional LSTM，hidden 包含以下内容：
        hidden[0]：第一层正向的最后隐藏状态。
        hidden[1]：第一层反向的最后隐藏状态。
        hidden[2]：第二层正向的最后隐藏状态。
        hidden[3]：第二层反向的最后隐藏状态。

        output: [batch, seq_len, hidden_size * num_directions]
        正向层：output[:, -1, :hidden_size] 确实是正向层的最后一个时间步的输出。
        反向层：output[:, 0, hidden_size:] 才是反向层的最后一个时间步的输出，因为反向层是从序列末尾往前处理的。
        """
        ouput, (hidden, _) = self.lstm(x)
        hidden = self.dropout2(hidden)
        last_layer_hidden = torch.cat(
            (hidden[-2], hidden[-1]), dim=-1)  # [batch, hidden_size * 2]
        avg_seq_output = torch.mean(ouput, dim=1)  # [batch, hidden_size * 2]
        max_seq_output, _ = torch.max(ouput, dim=1)  # [batch, hidden_size * 2]
        concat_hidden = torch.cat(
            (last_layer_hidden, avg_seq_output, max_seq_output), dim=-1)  # [batch, hidden_size * 6]
        output = self.fc1(concat_hidden)
        output = self.fc2(output)
        return output
```

基于以上结果，我们直接使用该模型在未进行预训练的情况下，对 seo `sg.csv` 关键词进行测试结果如下:
- 整个样本的准确率为 **57.50%**
- 测试集样本的准确率为 **56.09%**

基于以上结果，我们直接使用该模型对 seo `shopee.csv` 关键词进行预训练，然后在 `sg.csv` 测试集进行测试结果如下:
- 测试集样本的准确率为 **75.73%**
20/20 [20:34<00:00, 63.46s/it, test_acc=75.73%, train_acc=97.13%]


5. 使用 shopee keyword_5 进行训练的结果如下（模型结构 3）:
```json
{
    "vocab_size": 77,
    "embed_dim": 25,
    "hidden_size": 128,
    "num_classes": 26,
    "padding_idx": 0,
    "num_epochs": 5,
    "learning_rate": 0.01,
    "batch_size": 2048
}
```


```python
class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, output_size: int, padding_idx: int):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout1 = nn.Dropout(0.15)
        self.lstm = nn.LSTM(embed_dim, hidden_size,
                            batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)
        self.dropout2 = nn.Dropout(0.35)
        self.fc1 = nn.Linear(hidden_size * 6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout1(x)
        """
        hidden: [num_layers * num_directions, batch, hidden_size]

        对于 2 layers 和 bidirectional LSTM，hidden 包含以下内容：
        hidden[0]：第一层正向的最后隐藏状态。
        hidden[1]：第一层反向的最后隐藏状态。
        hidden[2]：第二层正向的最后隐藏状态。
        hidden[3]：第二层反向的最后隐藏状态。

        output: [batch, seq_len, hidden_size * num_directions]
        正向层：output[:, -1, :hidden_size] 确实是正向层的最后一个时间步的输出。
        反向层：output[:, 0, hidden_size:] 才是反向层的最后一个时间步的输出，因为反向层是从序列末尾往前处理的。
        """
        ouput, (hidden, _) = self.lstm(x)
        hidden = self.dropout2(hidden)
        last_layer_hidden = torch.cat(
            (hidden[-2], hidden[-1]), dim=-1)  # [batch, hidden_size * 2]
        avg_seq_output = torch.mean(ouput, dim=1)  # [batch, hidden_size * 2]
        max_seq_output, _ = torch.max(ouput, dim=1)  # [batch, hidden_size * 2]
        concat_hidden = torch.cat(
            (last_layer_hidden, avg_seq_output, max_seq_output), dim=-1)  # [batch, hidden_size * 6]
        output = self.fc1(concat_hidden)
        output = self.fc2(output)
        return output
```