""" 利用 model 提取图片特征向量, 并存储到 Milvus 中 """

import os

from models.resnet_34 import ResNet34FeatureExtractor
from client import client

extractor = ResNet34FeatureExtractor()

root = "./data"


def insert_vector(root: str):
    for dirpath, foldername, filenames in os.walk(root):
        for filename in filenames:
            if not filename.lower().endswith(".jpg"):
                continue
            filepath = os.path.join(dirpath, filename)
            image_embedding = extractor(filepath)
            client.insert(
                "image_embeddings",
                {"vector": image_embedding, "filename": filepath},
            )
