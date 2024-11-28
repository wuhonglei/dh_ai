import torch.nn as nn
from torchvision import transforms
from typing import Literal

# 创建一个字典来存储激活值
activations = {}


def get_activation(name):
    # 定义钩子函数
    def hook(model, input, output):
        activations[name] = output
    return hook


def register_hook(model):
    cnn_layers = model.cnn
    cnn_names = []
    for i, layer in enumerate(cnn_layers):
        if isinstance(layer, nn.Conv2d):
            name = f'conv_{i}'
            cnn_names.append(name)
            layer.register_forward_hook(get_activation(name))

    rnn_name = 'rnn'
    model.rnn[0].register_forward_hook(get_activation(rnn_name))
    return cnn_names, rnn_name


def get_transfrom_fn(in_channels: int, height: int, width: int, mode: Literal['training', 'evaluate']):
    transforms_list = []
    if in_channels == 1:
        transforms_list.append(transforms.Grayscale(num_output_channels=1))
    transforms_list.extend([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if mode == 'training':
        transforms_list.insert(-2, transforms.RandomRotation(10))

    return transforms.Compose(transforms_list)
