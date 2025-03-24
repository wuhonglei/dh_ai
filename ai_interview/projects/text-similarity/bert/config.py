from dynaconf import Dynaconf
from pydantic import BaseModel


class CacheConfig(BaseModel):
    search_history_path: str
    val_cbow_dataset_cache_path: str
    val_cbow_model_cache_path: str
    val_cbow_model_checkpoint_path: str


class DataSetConfig(BaseModel):
    train_csv_path: str
    val_csv_path: str


class VocabConfig(BaseModel):
    bert_name: str
    vocab_path: str
    embedding_dim: int
    max_length: int
    use_projection: bool


class MilvusConfig(BaseModel):
    version: str
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
    cache: CacheConfig
    dataset: DataSetConfig
    milvus: MilvusConfig
    evaluate_title: EvaluateTitleConfig
    vocab: VocabConfig


# 加载 Dynaconf 配置
_config = Dynaconf(
    settings_files=["config.toml"],
    lowercase_read=True,  # 添加这个参数，使读取时保持小写键名
)

# 处理版本配置
version = _config.milvus.version
_config.milvus.update(_config[version])

# 转换为带类型的配置对象
_config_dict = {k.lower(): v for k, v in _config.items()}  # 手动转换
config: AppConfig = AppConfig.model_validate(_config_dict)

# 导出为模块级变量，使其他模块可以直接导入
CACHE_CONFIG = config.cache
DATASET_CONFIG = config.dataset
MILVUS_CONFIG = config.milvus
EVALUATE_TITLE_CONFIG = config.evaluate_title
VOCAB_CONFIG = config.vocab
