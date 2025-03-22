"""
从 val.csv 中随机抽取1000条数据，保存为 标题测试原材料.csv
"""
import pandas as pd
import sys
from config import DATASET_CONFIG

# fmt: off
# sys.path.append('../')
# fmt: on

df = pd.read_csv(DATASET_CONFIG.val_csv_path)


df.sample(1000, random_state=42).to_csv('标题测试原材料.csv', index=False)
