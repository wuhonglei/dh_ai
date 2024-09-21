""" 利用 model 提取图片特征向量, 并存储到 Milvus 中 """

import os
import torch
from client import client
from models.resnet_34 import ResNet34FeatureExtractor
from models.similarity import process_img

extractor = ResNet34FeatureExtractor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor.to(device)


def insert_vector(root: str):
    total = 0
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

            extractor.eval()
            with torch.no_grad():
                image_embedding = extractor(img)
                image_embedding = image_embedding.detach().flatten(start_dim=1).cpu().numpy().tolist()[
                    0]

            client.insert(
                "image_embeddings",
                {"vector": image_embedding, "filename": filepath},
            )

    print(f"Total {total} images processed")


if __name__ == "__main__":
    root = "./data"
    insert_vector(root)
