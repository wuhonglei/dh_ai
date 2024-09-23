import os
from torch.utils.data import Dataset
from PIL import Image


class ImageReader(Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.abspath(root)
        self.transform = transform
        self.imgs = []
        for dirpath, foldername, filenames in os.walk(root):
            for filename in filenames:
                if not filename.lower().split('.')[-1] in ['jpg', 'jpeg', 'png']:
                    continue
                filepath = os.path.join(dirpath, filename)
                self.imgs.append(filepath)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = img
        return img_tensor, img_path.replace(self.root, ""), tuple((img.width, img.height))
