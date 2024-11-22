"""
图片数据的小批量加载
"""

import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from typing import Any, Tuple
import matplotlib.pyplot as plt
from utils import char_to_index
from typing import List


class CaptchaDataset(Dataset):
    def __init__(self, data_dir: str, padding_index, characters, captcha_length: int = -1,  transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.characters = characters
        self.imgs = os.listdir(data_dir)
        self.captcha_length = captcha_length
        self.padding_index = padding_index  # 标签长度不足时的填充字符

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = os.path.join(self.data_dir, self.imgs[idx])
        str_label = self.imgs[idx].split("_")[0]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        else:
            # 将 img 转为 tensor
            img = transforms.ToTensor()(img)

        # label_list = list(map(lambda char: char_to_index(
        #     char, self.characters, self.padding_index), str_label.lower()))
        # # if len(label_list) < self.captcha_length:
        # #     label_list += [int(self.padding_index)] * \
        # #         (self.captcha_length - len(label_list))

        # label = torch.tensor(label_list, dtype=torch.long)
        return (img, str_label.lower())


def encode_labels(labels: List[str], characters: str, padding_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将标签转换为索引序列和长度信息。
    Args:
        labels (list of str): 输入的标签字符串列表（如 ["abc", "12345"]）。
        characters (str): 包含所有可能字符的字符集。

    Returns:
        targets (Tensor): 拼接的索引序列，形状为 (total_length,)。
        target_lengths (Tensor): 每个标签的长度，形状为 (batch_size,)。
    """

    targets = []  # 保存所有标签的索引序列
    target_lengths = []  # 保存每个标签的长度

    for label in labels:
        indices = [char_to_index(char, characters, padding_index)
                   for char in label]  # 将每个字符映射为索引
        targets.extend(indices)  # 将当前标签的索引加入到 targets
        target_lengths.append(len(indices))  # 保存当前标签的长度

    # 转换为 PyTorch 张量
    targets = torch.LongTensor(targets)  # 所有标签拼接为一个张量
    target_lengths = torch.LongTensor(
        target_lengths)  # 每个标签的长度

    return targets, target_lengths


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])

    dataset = CaptchaDataset(
        data_dir='./data/train-3363-stable-new/test', characters='0123456789abcdefghijklmnopqrstuvwxyz', padding_index='36', captcha_length=6, transform=transform)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    def show_img(imgs: list[torch.Tensor], labels):
        import matplotlib.pyplot as plt
        total = len(imgs)
        print(f'total: {total}')
        for i in range(total):
            plt.subplot(2, total // 2, i + 1)
            img = imgs[i]
            label = labels[i]
            # label = ''.join(
            #     map(lambda x: x == int(padding_index), label.tolist()))
            img = img.numpy().transpose((1, 2, 0))
            plt.imshow(img)
            plt.axis('off')
            plt.title('label')

        plt.show()

    for epoch in range(1):
        print(f'epoch = {epoch}')
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            print('labels', labels)
            # show_img(imgs, labels)
            break
        print('---')
