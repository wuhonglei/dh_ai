from pymilvus import MilvusClient

client = MilvusClient(uri="example.db")
if client.has_collection(collection_name="image_embeddings"):
    client.drop_collection(collection_name="image_embeddings")

client.create_collection(
    collection_name="image_embeddings",
    vector_field_name="vector",
    dimension=512,
    auto_id=True,
    enable_dynamic_field=True,
    metric_type="COSINE",
)
