""" 利用 model 提取图片特征向量, 并存储到 Milvus 中 """
# fmt: off
import os
import time
import sys

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


def to_numpy(tensor):
    return tensor.detach().flatten(start_dim=1).cpu().numpy()


def insert_vector(root: str):
    total = 0
    client = MilvusClient(uri=os.path.join(current_dir,  "example.db"))
    init_client(client)

    start_time = time.time()
    time_dict = {
        'resnet34': 0.0,
        'vgg19': 0.0,
        'resnet34_insert': 0.0,
        'vgg19_insert': 0.0,
    }

    dataset = ImageReader(root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    for imgs, img_paths in dataloader:
        total += len(imgs)
        imgs = imgs.to(device)
        time1 = time.time()
        resnet34_image_embedding = resnet34_extractor(imgs)
        time2 = time.time()
        vgg19_image_embedding = vgg19_extractor(imgs)
        time3 = time.time()
        resnet34_image_embedding = to_numpy(resnet34_image_embedding)
        vgg19_image_embedding = to_numpy(vgg19_image_embedding)

        time4 = time.time()

        resnet34_datalist = []
        vgg19_datalist = []
        for i in range(len(img_paths)):
            resnet34_datalist.append({
                "vector": resnet34_image_embedding[i],
                "filename": img_paths[i]
            })
            vgg19_datalist.append({
                "vector": vgg19_image_embedding[i],
                "filename": img_paths[i]
            })

        client.insert(
            "resnet34",
            resnet34_datalist,
        )
        time5 = time.time()
        client.insert(
            "vgg19",
            vgg19_datalist,
        )
        time6 = time.time()

        time_dict['resnet34'] += time2 - time1
        time_dict['vgg19'] += time3 - time2
        time_dict['resnet34_insert'] += time5 - time4
        time_dict['vgg19_insert'] += time6 - time5
        print(f"Insert {total}/{len(dataset)} successfully")

    print(f"Total time: {time.time() - start_time}")
    print(f"Total {total} images processed")
    print(f"time_dict: {time_dict}")


if __name__ == "__main__":
    root = os.path.join(current_dir,  "data")
    insert_vector(root)
