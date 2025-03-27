from train import main
from utils.category import load_category_list
import pandas as pd

if __name__ == '__main__':
    train_df = pd.read_csv('./dataset/level1/train.csv',
                           usecols=['level1_global_be_category_id'])
    # 获取 level1_global_be_category_id 列的唯一值列表
    category_id_list = train_df['level1_global_be_category_id'].unique(
    ).tolist()

    main(data_dir='./dataset/level1/', label_names=['level1_global_be_category_id'],
         category_id_list=category_id_list, model_name='level1', tags=['level1'])
