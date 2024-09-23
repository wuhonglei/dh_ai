""" 利用 model 提取图片特征向量, 并存储到 Milvus 中 """

import os
import time

import torch
from pymilvus import MilvusClient
from models.resnet_34 import ResNet34FeatureExtractor
from models.vgg19_feature import VGG19FeatureExtractor
from models.similarity import process_img

resnet34_extractor = ResNet34FeatureExtractor()
vgg19_extractor = VGG19FeatureExtractor()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet34_extractor.to(device)
vgg19_extractor.to(device)
current_dir = os.path.dirname(os.path.abspath(__file__))


def init_client(client: MilvusClient) -> None:
    if client.has_collection(collection_name="resnet34"):
        client.drop_collection(collection_name="resnet34")

    client.create_collection(
        collection_name="resnet34",
        vector_field_name="vector",
        dimension=512,
        auto_id=True,
        enable_dynamic_field=True,
        metric_type="COSINE",
    )

    if client.has_collection(collection_name="vgg19"):
        client.drop_collection(collection_name="vgg19")

    client.create_collection(
        collection_name="vgg19",
        vector_field_name="vector",
        dimension=4096,
        auto_id=True,
        enable_dynamic_field=True,
        metric_type="COSINE",
    )


def to_numpy(tensor):
    return tensor.detach().flatten(start_dim=1).cpu().numpy()


def insert_vector(root: str):
    total = 0
    start_time = time.time()
    client = MilvusClient(uri=os.path.join(current_dir,  "example.db"))
    time1 = time.time()
    print(f"Connect to Milvus in {time1 - start_time:.2f}s")
    init_client(client)
    time2 = time.time()
    print(f"Init Milvus in {time2 - time1:.2f}s")

    return
    for dirpath, foldername, filenames in os.walk(root):
        print(f"Processing {dirpath}")
        for filename in filenames:
            # 文件名称以 jpg 或 jpeg 结尾
            if not filename.lower().split('.')[-1] in ['jpg', 'jpeg', 'png']:
                print(f"Skip {filename}")
                continue

            total += 1
            print(f"Processing {filename}")
            filepath = os.path.join(dirpath, filename)
            img = process_img(filepath)
            if img is None:
                print(f"Skip {filename}")
                continue

            print('filepath:', filepath)
            resnet34_image_embedding = resnet34_extractor(img)
            vgg19_image_embedding = vgg19_extractor(img)
            resnet34_image_embedding = to_numpy(resnet34_image_embedding)[0]
            vgg19_image_embedding = to_numpy(vgg19_image_embedding)[0]

            client.insert(
                "resnet34",
                {"vector": resnet34_image_embedding, "filename": filepath},
            )
            client.insert(
                "vgg19",
                {"vector": vgg19_image_embedding, "filename": filepath},
            )
            print(f"Insert {total}/1017 successfully")

    print(f"Total {total} images processed")


if __name__ == "__main__":
    root = os.path.join(current_dir,  "data")
    insert_vector(root)
