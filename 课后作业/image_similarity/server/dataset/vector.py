""" 利用 model 提取图片特征向量, 并存储到 Milvus 中 """
# fmt: off
import os
import time
import sys
import numpy as np

sys.path.append(os.getcwd())  # 将根目录导入模块搜索路径

import torch
from pymilvus import MilvusClient
from torch.utils.data import DataLoader

from models.resnet_34 import ResNet34FeatureExtractor
from models.vgg19_feature import VGG19FeatureExtractor
from models.similarity import process_img, transform

from image_reader import ImageReader

# fmt: on

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


def to_norm_numpy(input_tensor) -> np.ndarray[np.float32, np.dtype[np.float32]]:
    new_tensor = input_tensor.detach().flatten(start_dim=1)
    new_tensor = new_tensor / torch.norm(new_tensor, dim=1, keepdim=True)
    return new_tensor.cpu().numpy()


def insert_vector(root: str):
    total = 0
    client = MilvusClient(uri=os.path.join(current_dir,  "example.db"))
    init_client(client)
    start_time = time.time()
    dataset = ImageReader(root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    for (imgs, img_paths, img_sizes) in dataloader:
        imgs = imgs.to(device)

        resnet34_image_embedding = resnet34_extractor(imgs)
        vgg19_image_embedding = vgg19_extractor(imgs)

        resnet34_image_embedding = to_norm_numpy(resnet34_image_embedding)
        vgg19_image_embedding = to_norm_numpy(vgg19_image_embedding)

        resnet34_datalist = []
        vgg19_datalist = []
        for i in range(len(img_paths)):
            total += 1
            common = {
                "width": int(img_sizes[0][i].item()),
                "height": int(img_sizes[1][i].item()),
                "filename": img_paths[i],
            }
            resnet34_datalist.append({
                **common,
                "vector": resnet34_image_embedding[i],
            })
            vgg19_datalist.append({
                **common,
                "vector": vgg19_image_embedding[i],
            })
            print(f"Insert {total}/{len(dataset)}")

        client.insert(
            "resnet34",
            resnet34_datalist,
        )
        client.insert(
            "vgg19",
            vgg19_datalist,
        )

        print(f"Insert {total}/{len(dataset)} successfully")

    print(f"Total time: {time.time() - start_time}")
    print(f"Total {total} images processed")


if __name__ == "__main__":
    root = os.path.join(current_dir,  "data")
    insert_vector(root)
