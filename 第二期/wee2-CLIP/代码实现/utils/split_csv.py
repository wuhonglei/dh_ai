import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def split_csv(csv_path: str, test_size: float):
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42)
    return train_df, test_df


def save_csv(df: pd.DataFrame, path: str):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    df.to_csv(path, index=False)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "datasets" / "captions.csv"
    train_df, test_df = split_csv(str(csv_path), test_size=0.2)
    save_csv(train_df, str(project_root / "datasets" / "train_split.csv"))
    save_csv(test_df, str(project_root / "datasets" / "test_split.csv"))
