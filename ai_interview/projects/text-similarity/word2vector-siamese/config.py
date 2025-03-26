from dynaconf import Dynaconf
from pydantic import BaseModel
from typing import Optional


class CacheConfig(BaseModel):
    search_history_path: str
    best_model_path: str
    pre_trained_model_path: str


class DataSetConfig(BaseModel):
    train_csv_path: str  # 训练数据集
    val_csv_path: str  # 验证数据集
    test_csv_path: str  # 测试数据集


class VocabConfig(BaseModel):
    min_freq: int
    max_freq: int
    vocab_path: str
    use_pretrained_model: bool
    use_stop_words: bool
    embedding_dim: int
    projection_dim: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    temperature: float
    max_title_length: int
    max_content_length: int
    stop_words_paths: list[str]
    idf_path: str


class MilvusConfig(BaseModel):
    db_name: str
    collection_name: str
    description: str
    metric_type: str
    embedding_name: str


class EvaluateTitleConfig(BaseModel):
    limit: int
    test_data_path: str
    evaluate_result_path: str


class AppConfig(BaseModel):
    version: str
    project: str
    cache: CacheConfig
    dataset: DataSetConfig
    milvus: MilvusConfig
    evaluate_title: EvaluateTitleConfig
    vocab: VocabConfig


# 加载 Dynaconf 配置
_config = Dynaconf(
    settings_files=["config.toml"],
)

# 处理版本配置
version = _config.version
_config.milvus.update(_config[version])

# 转换为带类型的配置对象
_config_dict = {k.lower(): v for k, v in _config.items()}  # 手动转换
config: AppConfig = AppConfig.model_validate(_config_dict)

# 导出为模块级变量，使其他模块可以直接导入
VERSION = config.version
PROJECT = config.project
CACHE_CONFIG = config.cache
DATASET_CONFIG = config.dataset
MILVUS_CONFIG = config.milvus
EVALUATE_TITLE_CONFIG = config.evaluate_title
VOCAB_CONFIG = config.vocab
CONFIG = config
