from dynaconf import Dynaconf
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union, List


class CacheConfig(BaseModel):
    search_history_path: str


class DataConfig(BaseModel):
    vocab_path: str
    val_csv_path: str
    min_freq: int


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
    data: DataConfig
    milvus: MilvusConfig
    evaluate: EvaluateConfig


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
DATA_CONFIG = config.data
MILVUS_CONFIG = config.milvus
EVALUATE_CONFIG = config.evaluate
