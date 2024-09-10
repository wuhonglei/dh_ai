"""
一位或两位数字验证码识别
单独的模型识别率:
一位数字验证码识别率: 0.992
两位数字验证码识别率: 0.936


模型 1 和模型 2 识别率:
一位数字验证码识别率: 0.912
两位数字验证码识别率: 0.472
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CNNModel
from dataset import CaptchaDataset
from utils import load_model

model1 = load_model(captcha_length=1, class_num=10,
                    model_path='./models/1-model-dropout-more-nn.pth')
model2 = load_model(captcha_length=2, class_num=10,
                    model_path='./models/2-model_199.pth')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def avg_props(predict, probabilities):
    """
    计算平均概率
    :param predict: 预测结果
    :param probabilities: 概率
    """
    props = []
    for max_index, list_ in zip(predict, probabilities):
        props.append(list_[max_index])
    return sum(props) / len(props)


def list_to_str(list_):
    """
    将二维列表转为字符串
    :example: [[1, 2], [3, 4]] -> ['12', '34']
    """
    return [
        ''.join(map(str, x))
        for x in list_
    ]


def best_predict(model1, model2, imgs):
    """
    获取两个模型最佳的预测结果
    :param model1: 模型1
    :param model2: 模型2
    :param img: 图片
    """

    output = []
    for img in imgs:
        if model1 and not model2:
            output.append(model1.predict(img)[0][0].tolist())
            continue
        if model2 and not model1:
            output.append(model2.predict(img)[0][0].tolist())
            continue

        predict1, prob1 = model1.predict(img)
        avg_props1 = avg_props(predict1[0], prob1[0])
        predict2, prob2 = model2.predict(img)
        avg_props2 = avg_props(predict2[0], prob2[0])
        if avg_props1 > avg_props2:
            output.append(predict1[0].tolist())
        else:
            output.append(predict2[0].tolist())

    return list_to_str(output)


test_dataset = CaptchaDataset(data_dir='./data/多位验证码/两位', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
correct = 0
total = 0
for i, (img, label) in enumerate(test_loader):
    predict = best_predict(model1, model2, img)
    label = list_to_str(label.tolist())
    for p, l in zip(predict, label):
        if p == l:
            correct += 1
        total += 1

print(f'accuracy: {correct / total}')
