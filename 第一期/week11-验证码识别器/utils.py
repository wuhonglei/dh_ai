import os
import shutil
import yaml
import torch
from typing import List, Dict
from torch import nn
import json
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import matplotlib.pyplot as plt

import wandb


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
    preds = preds.cpu().numpy()  # type: ignore
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


def correct_predictions(preds: torch.Tensor, labels: List[str], characters: str, padding_index: int):
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

    return correct, decoded_preds


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


def rm_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)


def image_convert(chw_image):
    """ 将 (C, H, W) 转为 (H, W, C) """
    return chw_image.permute(1, 2, 0).cpu().numpy()


def de_transform(img_tensor):
    """ 反归一化 """
    return img_tensor * 0.5 + 0.5


def normalized_img(img_tensor):
    """ 归一化到 [0, 1] """
    min_val, max_val = img_tensor.min(), img_tensor.max()
    return (img_tensor - min_val) / (max_val - min_val)


def wandb_image(origin_eval_dataset, eval_dataset, batch_index: int, pred_label: str, true_label: str, wandb, table: wandb.Table):
    origin_image = image_convert(origin_eval_dataset[batch_index][0])
    transformed_image = image_convert(eval_dataset[batch_index][0])
    if pred_label != true_label:
        # 创建 wandb.Image 对象
        wandb_origin_image = wandb.Image(origin_image, mode='RGB')
        wandb_transformed_image = wandb.Image(
            transformed_image, mode='L')
        # 添加数据到表格
        table.add_data(wandb_origin_image,
                       wandb_transformed_image, pred_label, true_label)


def visualize_activations(origin_eval_dataset, eval_dataset, activations: Dict[str, torch.Tensor], cnn_names: List[str], rnn_name: str, batch_index: int, pred_label: str, true_label: str, visualize_all: bool) -> int:
    correct = pred_label == true_label
    if not visualize_all and correct:
        return 0

    img_name = origin_eval_dataset.imgs[batch_index]
    log_dir = f'logs/{img_name}'
    tag = f'({pred_label})'
    rm_dir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_image(
        'Image/Origin' + tag, origin_eval_dataset[batch_index][0], global_step=0)
    writer.add_image('Image/Transformed' + tag,
                     eval_dataset[batch_index][0], global_step=0)

    for cnn_name in cnn_names:
        conv_output = activations[cnn_name].detach()
        channel_num = conv_output.shape[1]

        writer.add_image(
            f'{cnn_name}/total_{channel_num}/Channel_0_mean', normalized_img(conv_output[0].mean(dim=0).unsqueeze(0)), global_step=0)
        writer.add_image(
            f'{cnn_name}/total_{channel_num}/Channel_0_max', normalized_img(conv_output[0].max(dim=0).values.unsqueeze(0)), global_step=0)

        for idx in range(channel_num):
            feature_map = conv_output[0, idx, :, :].unsqueeze(0)
            writer.add_image(
                f'{cnn_name}/total_{channel_num}/Channel_{idx + 1}', feature_map, global_step=0)

    # 可视化 RNN 输出
    rnn_output = activations[rnn_name]
    # 绘制并保存 RNN 输出热力图
    rnn_output_sample = rnn_output[0].squeeze(
        1).detach().cpu().numpy()  # [seq_len, hidden_size]
    # writer.add_histogram('RNN/Outputs', rnn_output_sample, global_step=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    cax = ax.imshow(rnn_output_sample.T, aspect='auto', cmap='hot')
    fig.colorbar(cax)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Hidden Units')
    ax.set_title('LSTM Output Heatmap')
    # 保存到 TensorBoard
    writer.add_figure('RNN/Heatmap' + tag, fig)
    plt.close(fig)

    # 累加每个时间步的 hidden_size, 并绘制柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    summed_output = rnn_output_sample.sum(axis=1)
    ax.bar(range(len(summed_output)), summed_output)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Sum of Hidden Units')
    ax.set_title('Sum of Hidden Units over Time')
    # 保存到 TensorBoard
    writer.add_figure('RNN/Sum of Hidden Units', fig)
    plt.close(fig)

    # writer.add_graph(model, imgs)
    writer.close()

    return 1


def get_tags_from_dir(dir_path_list: List[str]) -> List[str]:
    return [
        dir_path.replace('data/', '') for dir_path in dir_path_list
    ]


def load_json(file_path: str):
    print('loading json from', file_path)
    with open(file_path, 'r') as f:
        return json.load(f)
