import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

current_dir = Path(__file__).parent
parent_dir = current_dir.parent


@dataclass
class BowConfig:
    search_history_path: str = os.path.join(
        current_dir, "cache",  ".search_history")

    vocab_path: str = os.path.join(parent_dir, "data", "vocab.txt")
    val_csv_path: str = os.path.join(parent_dir, "data", "val.csv")
    min_freq: int = 1000

    db_name: str = os.path.join(current_dir, "database", "milvus_demo.db")
    collection_name: str = "bow_title_content_collection"
    metric_type: str = "COSINE"
    embedding_name: str = "content_embedding"
