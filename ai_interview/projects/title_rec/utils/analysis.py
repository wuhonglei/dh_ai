import torch


def analysis_gpu_memory(device: str):
    # 统计显存使用情况
    allocated_memory = torch.cuda.memory_allocated(
        device) / 1024**2  # 已分配显存（MB）
    reserved_memory = torch.cuda.memory_reserved(
        device) / 1024**2    # 已保留显存（MB）
    max_allocated_memory = torch.cuda.max_memory_allocated(
        device) / 1024**2  # 最大分配显存（MB）
    max_reserved_memory = torch.cuda.max_memory_reserved(
        device) / 1024**2   # 最大保留显存（MB）

    print(f"  Allocated Memory: {allocated_memory:.2f} MB")
    print(f"  Reserved Memory: {reserved_memory:.2f} MB")
    print(f"  Max Allocated Memory: {max_allocated_memory:.2f} MB")
    print(f"  Max Reserved Memory: {max_reserved_memory:.2f} MB")
