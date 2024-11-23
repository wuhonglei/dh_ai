import os
import shutil
import yaml
import torch
from typing import List
from torch import nn


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


def get_wandb_config():
    config = load_config()
    dataset_config = config['dataset']
    wandb_init_args = {
        'project': f'{config["name"]}-字符长度 {dataset_config["captcha_length"]}_{dataset_config["characters"]}',
        'config': config,
        'reinit': True
    }
    return wandb_init_args


def char_to_index(char: str, characters: str, padding_index: int) -> int:
    """
    字符转索引
    :param char: 字符
    :param characters: 字符集
    :return: 索引
    """
    index = characters.index(char)
    if len(str(padding_index)) == 0:
        return index

    if int(padding_index) == 0:
        return index + 1

    return index


def index_to_char(index: int, characters: str, padding_index: int) -> str:
    """
    索引转字符
    :param index: 索引
    :param characters: 字符集
    :return: 字符
    """
    if len(str(padding_index)) == 0:
        return characters[index]

    if int(padding_index) == 0:
        return characters[index - 1] if index > 0 else ''

    return characters[index]


def decode_predictions(preds: torch.Tensor, characters: str, padding_index: int) -> list[str]:
    # preds 的形状为 (seq_len, batch_size, num_classes)
    preds = preds.permute(1, 0, 2)  # 转为 (batch_size, seq_len, num_classes)
    preds = torch.argmax(preds, dim=-1)  # 在类别维度取最大值，得到索引
    preds = preds.cpu().numpy()
    decoded_results: list[str] = []
    for pred in preds:
        # 去除连续重复的索引和空白符（索引 0）
        chars: list[str] = []
        prev_idx = None
        for idx in pred:
            idx = int(idx)
            if idx != prev_idx and idx != padding_index:
                chars.append(index_to_char(idx, characters, padding_index))
            prev_idx = idx
        decoded_results.append(''.join(chars))
    return decoded_results


def correct_predictions(preds: torch.Tensor, labels: List[str], characters: str, padding_index: int) -> int:
    """
    计算正确预测的数量
    :param preds: 预测结果
    :param labels: 目标结果 ['123', 'abcd']
    :param padding_index: 填充索引
    :return: 正确预测的数量
    """
    correct = 0
    decoded_preds = decode_predictions(preds, characters, padding_index)
    for pred, label in zip(decoded_preds, labels):
        if pred == label:
            correct += 1

    return correct


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


def make_dir(filepath: str):
    base_dir = os.path.dirname(filepath)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)


def save_model(model_path: str, model: nn.Module):
    make_dir(model_path)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')


def clean_up(model_path, model):
    print('Cleaning up...')
    save_model(model_path, model)
