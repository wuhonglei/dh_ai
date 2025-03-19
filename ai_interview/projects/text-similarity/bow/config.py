from dynaconf import Dynaconf
from pydantic import BaseModel


class CacheConfig(BaseModel):
    search_history_path: str


class DataSetConfig(BaseModel):
    val_csv_path: str


class VocabConfig(BaseModel):
    min_freq: int
    vocab_path: str
    use_stop_words: bool
    stop_words_paths: list[str]


class MilvusConfig(BaseModel):
    version: str
    db_name: str
    collection_name: str
    description: str
    metric_type: str
    embedding_name: str


class EvaluateConfig(BaseModel):
    test_data_path: str
    evaluate_result_path: str


class AppConfig(BaseModel):
    cache: CacheConfig
    dataset: DataSetConfig
    milvus: MilvusConfig
    evaluate: EvaluateConfig
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
EVALUATE_CONFIG = config.evaluate
VOCAB_CONFIG = config.vocab
