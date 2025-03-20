from typing import List, TypedDict, Union, Optional
from numpy.typing import NDArray
import numpy as np
from config import MILVUS_CONFIG, MilvusConfig

from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus.milvus_client import IndexParams
from type_definitions import DataItem, DbResult


class MilvusDB:
    def __init__(self, dimension: int, milvus_config: Optional[MilvusConfig]):
        # 使用整个配置对象
        self.config = milvus_config or MILVUS_CONFIG
        self.client = MilvusClient(uri=self.config.db_name)
        self.dimension = dimension

        self.create_collection()

    def create_collection(self):
        try:
            if not self.client.has_collection(self.config.collection_name):
                fields = [
                    FieldSchema(name="index", dtype=DataType.INT64,
                                is_primary=True),
                    FieldSchema(name=self.config.embedding_name,
                                dtype=DataType.FLOAT16_VECTOR, dim=self.dimension),
                ]
                schema = CollectionSchema(
                    fields, self.config.description)
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    schema=schema,
                    index_params=IndexParams(
                        field_name=self.config.embedding_name,
                        index_type="FLAT",
                        metric_type=self.config.metric_type,
                        params={}
                    )
                )
                print(
                    f"Collection '{self.config.collection_name}' created with index.")
            else:
                print(
                    f"Collection '{self.config.collection_name}' already exists.")
        except Exception as e:
            print(f"Error creating collection: {e}")

    def get_collection(self):
        return self.client.load_collection(self.config.collection_name)

    def insert(self, data: Union[List[DataItem], DataItem]):
        self.client.insert(collection_name=self.config.collection_name,
                           data=data)  # type: ignore

    def search(self, embeddings: list[NDArray[np.float16]], limit: int = 10) -> list[list[DbResult]]:
        results = self.client.search(
            collection_name=self.config.collection_name,
            data=embeddings,
            limit=limit,
            anns_field=self.config.embedding_name,
            search_params={"metric_type": self.config.metric_type},
        )
        return results  # type: ignore


if __name__ == "__main__":
    df = MilvusDB(dimension=100, milvus_config=MILVUS_CONFIG)
