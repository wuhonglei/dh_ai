import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from typing import Union, IO

# from vgg16_feature import VGG16FeatureExtractor
# from vgg19_feature import VGG19FeatureExtractor


def compare_similarity(model, img1, img2):
    model.eval()
    with torch.no_grad():
        feature1 = model(img1)
        feature2 = model(img2)

    # Compute the similarity
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cosine_sim(feature1, feature2)
    return similarity.item()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def process_img(img_path: Union[str, IO[bytes]]) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # type: ignore
    return img


if __name__ == '__main__':
    pass
    # model = VGG16FeatureExtractor()
    # model = VGG19FeatureExtractor()
    # model.eval()

    # img1 = process_img('data/dog/3.png')
    # img2 = process_img('data/dog/4.png')

    # similarity = compare_similarity(model, img1, img2)
    # print(f'Similarity: {similarity}')
