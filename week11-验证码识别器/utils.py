import os
import shutil
import yaml

import torch
from model import CNNModel


def init_dir(dir_path, remove=False):
    """
    初始化目录
    :param dir_path: 目录路径
    :param remove: 是否删除已存在的目录
    """
    if remove and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def load_config(config_file='./config.yaml'):
    """
    加载配置文件
    :param config_file: 配置文件路径
    :return: 配置
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_wandb_config(captcha_length: int):
    config = load_config()
    wandb_init_args = {
        'project': f'{config["name"]}-字符长度 {captcha_length}_{config["dataset"]["characters"]}',
        'config': {
            "learning_rate": config['training']['learning_rate'],
            "architecture": f"{config['model']['type']}-{config['model']['layers']}",
            "epochs": config['training']['epochs'],
            "train_dataset_size": config['dataset']['train_total'],
            "test_dataset_size": config['dataset']['test_total'],
        },
        'reinit': True
    }
    return wandb_init_args


def load_model(captcha_length: int, class_num: int, model_path: str, input_size: int):
    """
    加载模型
    :param captcha_length: 验证码长度
    :param class_num: 类别数
    :param model_path: 模型路径
    :return: 模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(input_size=input_size, captcha_length=captcha_length,
                     class_num=class_num)
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True))
    return model


def char_to_index(char: str, characters: str):
    """
    字符转索引
    :param char: 字符
    :param characters: 字符集
    :return: 索引
    """
    return characters.index(char)


def get_max_length(captcha_length: str) -> int:
    """
    获取最大长度
    :param captcha_length: 验证码长度
    :return: 最大长度

    >>> get_max_length('1-5') == 5
    >>> get_max_length(4) == 4
    """
    return max(map(int, str(captcha_length).split('-')))


def get_len_range(captcha_length) -> tuple[int, int]:
    """
    获取长度范围
    :param captcha_length: 验证码长度
    :return: 长度范围

    >>> get_len_range('1-5') == [1, 5]
    >>> get_len_range(4) == [4, 4 + 1]
    """

    captcha_length = str(captcha_length)
    if '-' not in captcha_length:
        return (int(captcha_length), 1 + int(captcha_length))
    else:
        start, end = captcha_length.split('-')
        return (int(start), int(end) + 1)


class EarlyStopping:
    """ 早停策略 """

    def __init__(self, enable=True, patience=5, min_delta=0.0001):
        self.enable = enable
        self.patience = patience
        self.delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if not self.enable:
            return False

        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss > self.delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
