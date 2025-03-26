"""
从 val.csv 中随机抽取1000条数据，保存为 标题测试原材料.csv
"""
import os
import sys

import pandas as pd

# fmt: off
sys.path.append('../')
from config import DATASET_CONFIG
# fmt: on

df = pd.read_csv(os.path.join('..', DATASET_CONFIG.test_csv_path))

batch_size = 1000
df.sample(batch_size, random_state=42).to_csv(
    os.path.join('../data', f'test_{batch_size}.csv'), index=False)
