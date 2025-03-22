from model.cbow import CBOWModel
from config import DATASET_CONFIG, VOCAB_CONFIG, CACHE_CONFIG, MILVUS_CONFIG, config
from utils.common import get_device
from utils.train import is_enable_distributed, setup_distributed, cleanup_distributed, init_wandb
from cbow_dataset import CBOWDataset
from vocab import Vocab
from dataset import NewsDatasetCsv
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from typing import Union
import torch.distributed as dist


def save_model(model: CBOWModel, path: str):
    torch.save(model.state_dict(), path)


def build_loader(csv_dataset: NewsDatasetCsv, vocab: Vocab, window_size: int, batch_size: int, cache_path: str = ''):
    dataset = CBOWDataset(csv_dataset, vocab, window_size, cache_path)

    # 添加 DistributedSampler
    sampler = DistributedSampler(
        dataset) if is_enable_distributed() else None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler else True,
        sampler=sampler,
    )
    return dataloader, sampler


def evaluate(model: Union[CBOWModel, DDP], val_loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    local_batches = 0
    with torch.no_grad():
        for i, (context_idxs, target_idx) in enumerate(val_loader):
            context_idxs = context_idxs.to(device)
            target_idx = target_idx.to(device)
            loss = model(context_idxs, target_idx)
            total_loss += loss.item()
            local_batches += 1  # 当前进程的批次数

    if is_enable_distributed():
        # 同步总损失
        total_loss_tensor = torch.tensor(total_loss).to(device)
        dist.all_reduce(total_loss_tensor)
        global_total_loss = total_loss_tensor.item()

        # 同步全局批次数
        batch_tensor = torch.tensor(local_batches).to(device)
        dist.all_reduce(batch_tensor)
        global_batches = batch_tensor.item()

        # 返回全局平均批次损失
        return global_total_loss / global_batches

    return total_loss / len(val_loader)


def train():
    # 初始化分布式训练
    if is_enable_distributed():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        print(
            f"rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")
        setup_distributed(local_rank, rank, world_size)
    else:
        local_rank = 0
    device = get_device(is_enable_distributed(), local_rank)
    is_main_process = local_rank == 0

    epoch = 15
    batch_size = 1280
    learning_rate = 0.001

    vocab = Vocab(VOCAB_CONFIG)
    vocab.load_vocab_from_txt()
    vocab_size = len(vocab)
    window_size = VOCAB_CONFIG.window_size
    train_csv_dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    val_csv_dataset = NewsDatasetCsv(DATASET_CONFIG.test_csv_path)
    train_loader, train_sampler = build_loader(train_csv_dataset, vocab, window_size,
                                               batch_size, CACHE_CONFIG.val_cbow_dataset_cache_path)
    val_loader, val_sampler = build_loader(
        val_csv_dataset, vocab, window_size, batch_size)

    # 只在主进程初始化 wandb
    if is_main_process:
        init_wandb(config={
            **config.model_dump(),
            "train": {
                "epoch": epoch,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "vocab_size": vocab_size,
                "window_size": window_size,
            }
        })

    model = CBOWModel(vocab_size, VOCAB_CONFIG.embedding_dim, vocab.pad_idx)
    model = model.to(device)
    if os.path.exists(CACHE_CONFIG.val_cbow_model_cache_path):
        state_dict = torch.load(
            CACHE_CONFIG.val_cbow_model_cache_path,
            map_location=device
        )
        model.load_state_dict(state_dict)
        print(f"加载模型参数: {CACHE_CONFIG.val_cbow_model_cache_path}")

    origin_model = model
    if is_enable_distributed():
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01)  # 添加适当的权重衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    epoch_bar = tqdm(range(epoch), desc="训练",
                     disable=local_rank != 0, position=0)
    for epoch in epoch_bar:
        if train_sampler:
            train_sampler.set_epoch(epoch)
        total_loss = 0
        batch_bar = tqdm(
            train_loader, desc=f"训练第{epoch}轮", disable=local_rank != 0, position=1)
        val_loss = 0.0

        model.train()
        for i, (context_idxs, target_idx) in enumerate(batch_bar):
            context_idxs = context_idxs.to(device)
            target_idx = target_idx.to(device)
            optimizer.zero_grad()
            loss = model(context_idxs, target_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)  # 防止梯度爆炸
            optimizer.step()

            if is_main_process:  # 只在主进程记录日志
                batch_bar.set_postfix(loss=loss.item(), val_loss=val_loss)
                total_loss += loss.item()
                # 记录每个批次的损失
                wandb.log({"batch_loss": loss.item()},
                          step=epoch * len(train_loader) + i)

                if (i + 1) % 2000 == 0:
                    val_loss = evaluate(model, val_loader, device)
                    wandb.log({"val_loss": val_loss})

        # 在每个 epoch 末尾调用
        # scheduler.step()
        if is_main_process:
            """
            只在主进程记录 epoch 级别的指标
            在分布式环境下，total_loss 计算的是主进程的损失
            len(train_loader) 计算的主进程的批次数
            """
            avg_loss = total_loss / len(train_loader)
            epoch_bar.set_postfix(loss=avg_loss)
            # 记录每个 epoch 的平均损失
            wandb.log({"epoch": epoch, "avg_loss": avg_loss})
            save_model(origin_model, CACHE_CONFIG.val_cbow_model_checkpoint_path.replace(
                '.pth', f'_{epoch}.pth'))

    # 保存最终模型并关闭 wandb（只在主进程）
    if is_main_process:
        save_model(origin_model, CACHE_CONFIG.val_cbow_model_cache_path)
        wandb.finish()

    # 清理分布式进程组
    cleanup_distributed()


if __name__ == "__main__":
    train()
