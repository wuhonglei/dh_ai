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
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_wandb_config(captcha_length: int):
    config = load_config()
    wandb_init_args = {
        'project': f'{config["name"]}-字符长度 {captcha_length}',
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


def load_model(captcha_length: int, class_num: int, model_path: str):
    """
    加载模型
    :param captcha_length: 验证码长度
    :param class_num: 类别数
    :param model_path: 模型路径
    :return: 模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(captcha_length=captcha_length, class_num=class_num)
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True))
    return model
