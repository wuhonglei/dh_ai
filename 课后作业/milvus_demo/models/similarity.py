import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from typing import Union, IO


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


def process_img(img_path: Union[str, IO[bytes]]) -> Union[torch.Tensor, None]:
    try:
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0)  # type: ignore
        return img
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


if __name__ == '__main__':
    pass
