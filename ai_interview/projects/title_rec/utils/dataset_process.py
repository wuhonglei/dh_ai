import pandas as pd


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


if __name__ == "__main__":
    remove_columns()
