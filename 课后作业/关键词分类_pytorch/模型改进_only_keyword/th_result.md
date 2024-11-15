### 结果说明
![已有模型的训练结果](https://p.ipic.vip/9yq6rm.png)

#### 一、字符 LSTM 分类结果
1. 仅仅使用 SEO 搜索关键词进行训练的结果如下（模型结构 3）:
2024-11-15 11:41:11; epcoh: 14; test acc: 62.53%; train acc: 90.78%

```python
KeywordCategoryModel(
  (embedding): Embedding(50, 25, padding_idx=0)
  (dropout1): Dropout(p=0.15, inplace=False)
  (lstm): LSTM(25, 128, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
  (dropout2): Dropout(p=0.35, inplace=False)
  (fc1): Linear(in_features=768, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=26, bias=True)
)
```

```json
{
    "vocab_size": 102,
    "embed_dim": 51,
    "hidden_size": 128,
    "num_classes": 34,
    "padding_idx": 0,
    "num_epochs": 15,
    "learning_rate": 0.01,
    "batch_size": 2048,
    "vocab_cache": "./cache/vocab/TH_vocab_36603_dc81562acda9a2073d477e9a813bcc6c_100.json",
    "load_state_dict": "./models/weights/TH/TH_LSTM_128*2_fc_2_seo_1731664004_12.pth",
    "save_model": "TH_LSTM_128*2_fc_2_seo_1731669164",
    "log_file": "./logs/TH_1731669164.txt"
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