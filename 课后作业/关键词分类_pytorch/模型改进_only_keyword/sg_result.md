### 结果说明
![已有模型的训练结果](https://p.ipic.vip/9yq6rm.png)

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

```txt
2024-11-13 16:51:50; epcoh: 1; test acc: 73.48%; train acc: 73.79%
2024-11-13 17:31:50; epcoh: 2; test acc: 74.36%; train acc: 75.09%
2024-11-13 18:12:05; epcoh: 3; test acc: 74.80%; train acc: 75.28%
2024-11-13 18:51:32; epcoh: 4; test acc: 75.62%; train acc: 76.21%
2024-11-13 19:31:00; epcoh: 5; test acc: 75.67%; train acc: 76.33%
2024-11-13 20:10:37; epcoh: 6; test acc: 75.91%; train acc: 76.51%
```

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

基于以上结果，我们直接使用该模型在未进行预训练的情况下，对 seo `sg.csv` 关键词进行测试结果如下:
- 整个样本的准确率为 **58.15%**
- 测试集样本的准确率为 **57.91%**

基于以上结果，我们直接使用该模型对 seo `shopee.csv` 关键词进行预训练，然后在 `sg.csv` 测试集进行测试结果如下:
- 测试集样本的准确率为 **72.91%**
20/20 [22:55<00:00, 68.76s/it, test_acc=72.91%, train_acc=82.75%]

6. 使用 shopee keyword_10 进行训练的结果如下（模型结构 3）:

词汇表: `SG_vocab_852663_ed7981fe7082fd991eeb420a89f6c9b5.json`

```txt
2024-11-14 22:22:49; epcoh: 10; test acc: 87.24%; train acc: 88.57%
2024-11-14 22:54:37; epcoh: 11; test acc: 87.26%; train acc: empty
2024-11-14 23:50:51; epcoh: 12; test acc: 86.92%; train acc: 88.56%
2024-11-15 00:22:57; epcoh: 13; test acc: 87.33%; train acc: empty
2024-11-15 01:07:14; epcoh: 14; test acc: 87.34%; train acc: 88.84%
2024-11-15 01:39:31; epcoh: 15; test acc: 87.38%; train acc: empty
```

```json
{
    "vocab_size": 81,
    "embed_dim": 40,
    "hidden_size": 128,
    "num_classes": 26,
    "padding_idx": 0,
    "num_epochs": 15,
    "learning_rate": 0.01,
    "batch_size": 2048,
    "save_model": "SG_LSTM_128*2_fc_2_shopee_keyword_10_model_seo_1731598700",
    "log_file": "./logs/SG_1731598700.txt"
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
- 整个样本的准确率为 **60.47%**
- 测试集样本的准确率为 **59.27%**

基于以上结果，我们直接使用该模型对 seo `shopee.csv` 关键词进行预训练，然后在 `sg.csv` 测试集进行测试结果如下:
- 测试集样本的准确率为 **76.73**
日志文件地址: `SG_1731659994.txt`
```
2024-11-15 09:11:44; epcoh: 28; test acc: 77.91%; train acc: 98.07%
2024-11-15 09:12:32; epcoh: 29; test acc: 77.36%; train acc: empty
2024-11-15 09:17:00; epcoh: 30; test acc: 76.73%; train acc: 98.33%
```

7. 使用单词的 tf-idf + SEO 搜索关键词字符 进行训练的结果如下（模型结构 3）:
2024-11-15 16:48:13; epcoh: 14; test acc: 75.27%; train acc: 97.89%

```json
{
    "vocab_size": 42,
    "embed_dim": 25,
    "hidden_size": 128,
    "num_classes": 26,
    "padding_idx": 0,
    "num_epochs": 15,
    "learning_rate": 0.01,
    "batch_size": 2048,
    "vocab_cache": "./cache/vocab/SG_vocab_20887_cf772552a09fa5ed08054ba5a92b104b_100.json",
    "tf_idf_dim": 10662,
    "save_model": "SG_LSTM_128*2_fc_2_seo_1731688052",
    "log_file": "./logs/SG_1731688052.txt"
}
```

```python
class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, tf_idf_dim: int, output_size: int, padding_idx: int):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout1 = nn.Dropout(0.15)
        self.lstm = nn.LSTM(embed_dim, hidden_size,
                            batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)
        self.dropout2 = nn.Dropout(0.35)
        self.fc1 = nn.Linear(tf_idf_dim + hidden_size * 6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, tf_idf_vectors):
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
            (tf_idf_vectors, last_layer_hidden, avg_seq_output, max_seq_output), dim=-1)  # [batch, hidden_size * 6]
        output = self.fc1(concat_hidden)
        output = self.fc2(output)
        return output
```

8. 使用 imp_level1_category_1d + SEO 搜索关键词字符 进行训练的结果如下（模型结构 3）:
2024-11-15 17:20:10; epcoh: 12; test acc: 82.36%; train acc: 86.05%

```json
{
    "vocab_size": 44,
    "embed_dim": 25,
    "hidden_size": 128,
    "num_classes": 26,
    "padding_idx": 0,
    "num_epochs": 15,
    "learning_rate": 0.01,
    "batch_size": 2048,
    "vocab_cache": "./cache/vocab/SG_vocab_17229_54ab2d7562cd0d9179c0f6b266d6b3af_10.json",
    "sub_category": 49,
    "save_model": "SG_LSTM_128*2_fc_2_seo_1731690373",
    "log_file": "./logs/SG_1731690373.txt"
}
```

```python
class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, sub_category: int, output_size: int, padding_idx: int):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout1 = nn.Dropout(0.15)
        self.lstm = nn.LSTM(embed_dim, hidden_size,
                            batch_first=True, bidirectional=True, num_layers=2, dropout=0.25)
        self.dropout2 = nn.Dropout(0.35)
        self.fc1 = nn.Linear(sub_category + hidden_size * 6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, tf_idf_vectors):
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
            (tf_idf_vectors, last_layer_hidden, avg_seq_output, max_seq_output), dim=-1)  # [batch, hidden_size * 6]
        output = self.fc1(concat_hidden)
        output = self.fc2(output)
        return output

```


#### 二、字符 RNN 分类结果
1. 仅仅使用 SEO 搜索关键词进行训练的结果如下（模型结构 1）:
2024-11-14 23:38:32; epcoh: 20; test acc: 57.73%; train acc: 71.94%

```
KeywordCategoryModel(
  (embedding): Embedding(26, 25, padding_idx=0)
  (dropout1): Dropout(p=0.15, inplace=False)
  (rnn): RNN(25, 128, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
  (dropout2): Dropout(p=0.35, inplace=False)
  (fc): Linear(in_features=256, out_features=26, bias=True)
)
```

```json
{
    "vocab_size": 26,
    "embed_dim": 25,
    "hidden_size": 128,
    "num_classes": 26,
    "padding_idx": 0,
    "num_epochs": 20,
    "learning_rate": 0.01,
    "batch_size": 2048,
    "save_model": "SG_LSTM_128*2_fc_2_shopee_keyword_5_model_seo_1731597477",
    "log_file": "./logs/SG_1731597477.txt"
}
```

```python
class KeywordCategoryModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int, output_size: int, padding_idx: int, dropout: float = 0):
        super(KeywordCategoryModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout1 = nn.Dropout(0.15)
        self.rnn = nn.RNN(embed_dim,
                          hidden_size,
                          num_layers=2,
                          batch_first=True,
                          bidirectional=True,
                          dropout=0.25
                          )
        self.dropout2 = nn.Dropout(0.35)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = self.embedding(x)
        # hidden: [num_layers * num_directions, batch, hidden_size]
        _, hidden = self.rnn(x)
        last_layer_hidden = torch.cat(
            (hidden[-2], hidden[-1]), dim=-1)  # [batch, hidden_size * 2]
        x = self.dropout2(last_layer_hidden)
        output = self.fc(x)
        return output
```