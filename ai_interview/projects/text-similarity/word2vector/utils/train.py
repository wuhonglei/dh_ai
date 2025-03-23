import os
import torch
import torch.distributed as dist
import wandb

# fmt: off
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from type_definitions import Hyperparameters
# fmt: on


def is_enable_distributed():
    """
    判断当前是否启用了分布式环境
    """
    return "WORLD_SIZE" in os.environ


def setup_distributed(local_rank, rank, world_size):
    """
    设置分布式训练环境
    """
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 设置当前设备
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    """
    清理分布式训练环境
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def init_wandb(config: dict):
    wandb.init(
        project="text-similarity-word2vec_v2",
        config=config
    )


def get_hyperparameters(is_main_process: bool, device: torch.device):
    # 创建一个字典来存储超参数
    if is_main_process:
        # 主进程从 wandb.config 获取参数
        hyperparams = {
            'min_freq': wandb.config.min_freq,
            'max_freq': wandb.config.max_freq,
            'embedding_dim': wandb.config.embedding_dim,
            'batch_size': wandb.config.batch_size,
            'learning_rate': wandb.config.learning_rate,
            'weight_decay': wandb.config.weight_decay,
            'epochs': wandb.config.epochs,
            'window_size': wandb.config.window_size,
        }
        # 将字典转换为tensor
        hyperparam_tensor = torch.tensor([
            hyperparams['min_freq'],
            hyperparams['max_freq'],
            hyperparams['embedding_dim'],
            hyperparams['batch_size'],
            hyperparams['learning_rate'],
            hyperparams['weight_decay'],
            hyperparams['epochs'],
            hyperparams['window_size'],
        ], device=device)
    else:
        # 非主进程创建空tensor来接收数据
        hyperparam_tensor = torch.zeros(4, device=device)

    # 广播超参数
    if is_enable_distributed():
        dist.broadcast(hyperparam_tensor, src=0)

    # 将 tensor 转回字典
    hyperparams: Hyperparameters = {
        'min_freq': int(hyperparam_tensor[0].item()),
        'max_freq': int(hyperparam_tensor[1].item()),
        'embedding_dim': int(hyperparam_tensor[2].item()),
        'batch_size': int(hyperparam_tensor[3].item()),
        'learning_rate': float(hyperparam_tensor[4].item()),
        'weight_decay': float(hyperparam_tensor[5].item()),
        'epochs': int(hyperparam_tensor[6].item()),
        'window_size': int(hyperparam_tensor[7].item()),
    }

    return hyperparams


def get_checkpoint_path(hyperparams: Hyperparameters, epoch: int):
    return f'checkpoint/{hyperparams["min_freq"]}_{hyperparams["max_freq"]}_{hyperparams["window_size"]}_{hyperparams["epochs"]}_{hyperparams["learning_rate"]}_{hyperparams["weight_decay"]}_{hyperparams["embedding_dim"]}_{hyperparams["batch_size"]}_{epoch}.pth'


def get_checkpoint_path_final(hyperparams: Hyperparameters):
    return f'checkpoint/{hyperparams["min_freq"]}_{hyperparams["max_freq"]}_{hyperparams["window_size"]}_{hyperparams["epochs"]}_{hyperparams["learning_rate"]}_{hyperparams["weight_decay"]}_{hyperparams["embedding_dim"]}_{hyperparams["batch_size"]}_final.pth'


def get_train_dataset_cache_path(min_freq: int, max_freq: int, window_size: int):
    return f'cache/train_dataset_cache_{min_freq}_{max_freq}_{window_size}.pkl'
