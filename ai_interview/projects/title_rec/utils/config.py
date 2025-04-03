from pathlib import Path
import os

root_dir = str(Path(__file__).parent.parent)
dataset_dir = os.path.join(root_dir, 'dataset', 'level1_20')
clean_dataset_dir = os.path.join(dataset_dir, 'clean')
