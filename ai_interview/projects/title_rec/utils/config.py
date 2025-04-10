from pathlib import Path
import os

root_dir = str(Path(__file__).parent.parent)
dataset_dir = os.path.join(root_dir, 'dataset', 'level1_80')
clean_dataset_dir = os.path.join(dataset_dir, 'clean')
fasttext_dataset_dir = os.path.join(
    root_dir, 'models/fasttext_model/data')
leaf_level_dir = os.path.join(root_dir, 'models/leaf_level')
fasttext_leaf_level_dir = os.path.join(
    leaf_level_dir, 'fasttext_model')
fasttext_leaf_level_cascade_dataset_dir = os.path.join(
    fasttext_leaf_level_dir, 'cascade/data')
fasttext_leaf_level_joint_dataset_dir = os.path.join(
    fasttext_leaf_level_dir, 'joint/data')
