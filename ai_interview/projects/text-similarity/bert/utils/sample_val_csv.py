"""
从 val.csv 中随机抽取1000条数据，保存为 标题测试原材料.csv
"""
import os
import sys

import pandas as pd

# fmt: off
sys.path.append('../')
# fmt: on

src_csv_path = '../data/val_10000.csv'
src_csv_name = os.path.basename(src_csv_path).split('.')[0]
df = pd.read_csv(src_csv_path)

batch_size = 1000
df.sample(batch_size, random_state=42).to_csv(
    f'../data/evaluate/标题测试原材料_{batch_size}_from_{src_csv_name}.csv'.format(batch_size, src_csv_name), index=False)
