from typing import List, TypedDict, Union
from numpy.typing import NDArray
import numpy as np
from config import BowConfig

from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus.milvus_client import IndexParams


class DataItem(TypedDict):
    index: int
    content_embedding: NDArray[np.float16]


class SearchResult(TypedDict):
    id: int
    distance: float


class MilvusDB:
    def __init__(self, db_name: str, collection_name: str, dimension: int):
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = MilvusClient(uri=db_name)
        self.dimension = dimension
        self.create_collection()

    def create_collection(self):
        try:
            if not self.client.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name="index", dtype=DataType.INT64,
                                is_primary=True),
                    FieldSchema(name=BowConfig.embedding_name,
                                dtype=DataType.FLOAT16_VECTOR, dim=self.dimension),
                ]
                schema = CollectionSchema(
                    fields, "title_embedding and content_embedding")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=IndexParams(
                        field_name=BowConfig.embedding_name,
                        index_type="FLAT",
                        metric_type=BowConfig.metric_type,
                        params={}
                    )
                )
                print(
                    f"Collection '{self.collection_name}' created with index.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            print(f"Error creating collection: {e}")

    def get_collection(self):
        return self.client.load_collection(self.collection_name)

    def insert(self, data: Union[List[DataItem], DataItem]):
        self.client.insert(collection_name=self.collection_name,
                           data=data)  # type: ignore

    def search(self, embedding: NDArray[np.float16], limit: int = 10) -> List[SearchResult]:
        results = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=limit,
            anns_field=BowConfig.embedding_name,
            search_params={"metric_type": BowConfig.metric_type},
        )
        return results[0]  # type: ignore


if __name__ == "__main__":
    df = MilvusDB(db_name="./database/milvus_demo.db",
                  collection_name="bow_title_content_collection", dimension=100)
