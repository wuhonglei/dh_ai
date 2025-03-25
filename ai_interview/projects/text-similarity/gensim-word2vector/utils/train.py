import os
import torch
import torch.distributed as dist
import wandb
from config import WandbConfig


def is_enable_distributed():
    """
    判断当前是否启用了分布式环境
    """
    return "WORLD_SIZE" in os.environ


def setup_distributed(local_rank):
    """
    设置分布式训练环境
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

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


def get_checkpoint_path(hyperparams: WandbConfig, epoch: int):
    return f'checkpoint/{hyperparams.min_freq}_{hyperparams.window}_{hyperparams.epochs}_{hyperparams.embedding_dim}_{epoch}.pth'


def get_checkpoint_path_final(hyperparams: WandbConfig):
    return f'checkpoint/{hyperparams.min_freq}_{hyperparams.window}_{hyperparams.epochs}_{hyperparams.embedding_dim}_final.pth'


def get_train_dataset_cache_path():
    return f'cache/train_dataset_cache.pkl'
