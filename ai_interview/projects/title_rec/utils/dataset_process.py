import pandas as pd
import glob
import os
from tqdm import tqdm
from config import dataset_dir, clean_dataset_dir


def remove_columns():
    """
    删除数据集中的item_id, shop_id, clean_name列
    """

    csv_paths = [
        "dataset/level1/train.csv",
        "dataset/level1/val.csv",
        "dataset/level1/test.csv",
    ]
    save_paths = [
        "dataset/origin/train.csv",
        "dataset/origin/val.csv",
        "dataset/origin/test.csv",
    ]
    for csv_path, save_path in zip(csv_paths, save_paths):
        df = pd.read_csv(csv_path)
        df = df.drop(columns=["item_id", "shop_id", "clean_name"])
        df.to_csv(save_path, index=False)


def join_csv():
    csv_dir = dataset_dir
    output_dir = clean_dataset_dir
    input_csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    columns = ['main_image', 'attached_images',
               'download_success', 'main_image_name']
    for csv_file in tqdm(input_csv_files, desc='join csv'):
        csv_name = os.path.basename(csv_file)
        src_df = pd.read_csv(csv_file)
        dest_df = pd.read_csv(os.path.join(output_dir, csv_name))
        src_df.set_index('item_id', inplace=True)
        dest_df.set_index('item_id', inplace=True)
        for index, row in tqdm(dest_df.iterrows(), desc=f'join csv {csv_name}', total=len(dest_df)):
            if index in src_df.index:
                for column in columns:
                    dest_df.loc[index, column] = src_df.loc[index, column]
        dest_df.to_csv(os.path.join(output_dir, csv_name), index=True)


if __name__ == "__main__":
    join_csv()
