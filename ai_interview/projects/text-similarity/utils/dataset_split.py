import pandas as pd

df = pd.read_csv('../data/origin/sohu_data.csv')
len = len(df)

# 安装 0.7, 0.15, 0.15 的比例划分数据集
train_len = int(len * 0.7)
val_len = int(len * 0.15)
test_len = len - train_len - val_len

# 划分数据集
train_df = df.iloc[:train_len]
val_df = df.iloc[train_len:train_len + val_len]
test_df = df.iloc[train_len + val_len:]

# 保存数据集
train_df.to_csv('../data/train.csv', index=False)
val_df.to_csv('../data/val.csv', index=False)
test_df.to_csv('../data/test.csv', index=False)
