"""
从 val.csv 中随机抽取1000条数据，保存为 标题测试原材料.csv
"""
import os
import sys

import pandas as pd

# fmt: off
sys.path.append('../')
# fmt: on

df = pd.read_csv('../data/val.csv')

batch_size = 1000
df.sample(batch_size, random_state=42).to_csv(
    '../data/val_1000.csv', index=False)
