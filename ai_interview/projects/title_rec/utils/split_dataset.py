import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from tqdm import tqdm

from common import create_dir

root_dir = Path(__file__).parent.parent
random_state = 42


def get_csv_files(dir: str):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            yield os.path.join(root, file)


def merge_csv_files(filepaths: list[str]):
    """
    合并多个csv文件，并保存为dataset.csv
    """
    df = pd.concat([pd.read_csv(filepath) for filepath in filepaths])
    output_file = os.path.join(
        root_dir, "dataset", "dataset.csv")
    df.to_csv(output_file, index=False)
    return df


def split_dataset(df: pd.DataFrame, stratify_column: str):
    """
    将数据集按比例分割为 训练集 0.7,验证集 0.2, 测试集 0.1
    Args:
        df: 输入的数据框
        stratify_column: 用于分层采样的列名
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=random_state, stratify=df[stratify_column])
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.3, random_state=random_state, stratify=temp_df[stratify_column])
    return train_df, valid_df, test_df


def save_dataset():
    """
    保存数据集到指定路径
    """
    df = pd.read_csv(root_dir / "dataset" / "dataset.csv")
    stratify_column = "level1_global_be_category_id"

    # 数据验证
    if stratify_column not in df.columns:
        raise ValueError(f"列 {stratify_column} 不存在于数据集中")

    # 检查缺失值
    if df[stratify_column].isna().any():
        print(f"警告：{stratify_column} 列中存在缺失值，将删除这些行")
        df = df.dropna(subset=[stratify_column])

    # 检查类别分布
    class_counts = df[stratify_column].value_counts()
    print("\n类别分布:")
    print(class_counts)

    # 检查是否有样本量过少的类别
    min_samples = 2  # 每个类别至少需要的样本数
    small_classes = class_counts[class_counts < min_samples]
    if not small_classes.empty:
        print(f"\n警告：以下类别的样本数少于 {min_samples} 个:")
        print(small_classes)
        print("这些类别可能无法正确进行分层采样")

    # 确保标签列是分类类型
    df[stratify_column] = df[stratify_column].astype('category')

    # 进行分层采样
    sub_df, _ = train_test_split(
        df,
        test_size=0.2,
        random_state=random_state,
        stratify=df[stratify_column]
    )

    # 数据集分割
    train_df, valid_df, test_df = split_dataset(
        sub_df, stratify_column=stratify_column)

    # 创建输出目录
    output_dir = root_dir / "dataset" / "level1_20"
    create_dir(str(output_dir))

    # 保存数据集
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # 打印数据集信息
    total = len(sub_df)
    print("\n数据集分割完成:")
    print(f"训练集大小: {len(train_df)/total:.2%}")
    print(f"验证集大小: {len(valid_df)/total:.2%}")
    print(f"测试集大小: {len(test_df)/total:.2%}")
    print(f"数据集保存在: {output_dir}")


if __name__ == "__main__":
    save_dataset()
