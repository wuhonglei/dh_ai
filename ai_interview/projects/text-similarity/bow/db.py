from typing import List, TypedDict, Union
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import time


class DataItem(TypedDict):
    id: int
    title_embedding: List[float]
    content_embedding: List[float]


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
                    FieldSchema(name="id", dtype=DataType.INT64,
                                is_primary=True),
                    FieldSchema(name="index", dtype=DataType.INT64,
                                is_primary=False),
                    FieldSchema(name="title_embedding",
                                dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                    FieldSchema(name="content_embedding",
                                dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                ]
                schema = CollectionSchema(
                    fields, "title_embedding and content_embedding")
                self.client.create_collection(
                    collection_name=self.collection_name, schema=schema)
                print(f"Collection '{self.collection_name}' created.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            print(f"Error creating collection: {e}")

    def insert(self, data: Union[List[DataItem], DataItem]):
        self.client.insert(collection_name=self.collection_name,
                           data=data)  # type: ignore

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    with MilvusDB(db_name="./database/milvus_demo.db",
                  collection_name="bow_title_content_collection", dimension=100) as db:
        pass
