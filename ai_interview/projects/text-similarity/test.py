from pymilvus import MilvusClient, utility, Collection

client = MilvusClient("milvus_demo.db")


# 检查集合是否存在
collection_name = "demo_collection"
if not utility.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        dimension=5
    )
    print(f"Collection '{collection_name}' created successfully!")
else:
    print(f"Collection '{collection_name}' already exists, skipping creation.")
    # collection = Collection(name=collection_name)  # 直接加载已有集合
