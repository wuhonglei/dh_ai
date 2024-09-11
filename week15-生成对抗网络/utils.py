import os
import shutil


def make_dirs(path: str, remove: bool = False):
    if remove and os.path.exists(path):
        shutil.rmtree(path)  # 空目录

    os.makedirs(path, exist_ok=True)
