import os
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class AnimateDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data = []
        for image in os.listdir(root):
            if image.endswith('.jpg') or image.endswith('.png'):
                self.data.append(os.path.join(root, image))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = Image.open(self.data[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = AnimateDataset('demo', transform)

    # 显示图片
    img = dataset[0]
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.show()
