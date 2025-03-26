import os
import torch
import torch.distributed as dist
import wandb

# fmt: off
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from type_definitions import WandbConfig
# fmt: on


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


def get_best_checkpoint_path(hyperparams: WandbConfig):
    return f'checkpoint1/{hyperparams.min_freq}_{hyperparams.max_freq}_{hyperparams.embedding_dim}_{hyperparams.projection_dim}_{hyperparams.batch_size}_{hyperparams.learning_rate}_{hyperparams.weight_decay}_{hyperparams.epochs}_{hyperparams.temperature}_best.pth'


def get_checkpoint_path_final(hyperparams: WandbConfig):
    return f'checkpoint1/{hyperparams.min_freq}_{hyperparams.max_freq}_{hyperparams.embedding_dim}_{hyperparams.projection_dim}_{hyperparams.batch_size}_{hyperparams.learning_rate}_{hyperparams.weight_decay}_{hyperparams.epochs}_{hyperparams.temperature}_final.pth'
