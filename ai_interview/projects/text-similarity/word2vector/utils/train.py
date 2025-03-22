import os
import torch
import torch.distributed as dist


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
