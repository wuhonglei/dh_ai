import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import albumentations as A


class ImageDataset(Dataset):
    def __init__(self, image_names: list[str], image_dir: str, labels: np.ndarray, transform: A.Compose):
        self.image_names = image_names
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        image = augmented['image']
        label = self.labels[index]
        return image, label
