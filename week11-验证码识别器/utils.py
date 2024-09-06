import os
import shutil
import yaml


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


def get_wandb_config():
    config = load_config()
    wandb_init_args = {
        'project': f'{config["name"]}-字符长度 {config["dataset"]["captcha_length"]}',
        'config': {
            "learning_rate": config['training']['learning_rate'],
            "architecture": f"{config['model']['type']}-{config['model']['layers']}",
            "epochs": config['training']['epochs'],
        },
        'reinit': True
    }
    return wandb_init_args
