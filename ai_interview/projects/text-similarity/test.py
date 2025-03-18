from operator import index
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from pymilvus import MilvusClient, utility, Collection
from pymilvus.milvus_client import IndexParams

client = MilvusClient("milvus_demo.db")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64,
                is_primary=True),
    FieldSchema(name='content_embedding',
                dtype=DataType.FLOAT_VECTOR, dim=5),
    FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name='subject', dtype=DataType.VARCHAR, max_length=100),
]
schema = CollectionSchema(
    fields, "title_embedding and content_embedding", metric_type="COSINE")

client.create_collection(
    collection_name="demo_collection",
    schema=schema,
    index_params=IndexParams(field_name="content_embedding")
)

res = client.insert(
    collection_name="demo_collection",
    data=[
        {"id": 0, "content_embedding": [0.1, 0.2, 0.3, 0.2, 0.1],
         "text": "AI was proposed in 1956.", "subject": "history"},
        {"id": 1, "content_embedding": [0.1, 0.2, 0.3, 0.2, 0.1],
            "text": "Alan Turing was born in London.", "subject": "history"},
        {"id": 2, "content_embedding": [0.1, 0.2, 0.3, 0.2, 0.1],
            "text": "Computational synthesis with AI algorithms predicts molecular properties.", "subject": "biology"},
    ]
)

res = client.search(
    collection_name="demo_collection",
    data=[[0.1, 0.2, 0.3, 0.2, 0.1]],
    # filter="subject == 'history'",
    limit=2,
    output_fields=["text", "subject"],
)

print(res)
