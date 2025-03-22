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
from torch.nn.utils.rnn import pad_sequence
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import wandb


def collate_fn(batch: list[tuple[list[int], int]],  pad_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    context_idxs, target_idx = zip(*batch)
    context_idxs_tensor = [torch.tensor(idx, dtype=torch.long)
                           for idx in context_idxs]
    new_context_idxs_tensor = pad_sequence(
        context_idxs_tensor, padding_value=pad_idx, batch_first=True)
    return new_context_idxs_tensor, torch.tensor(target_idx, dtype=torch.long)


def save_model(model: CBOWModel, path: str):
    torch.save(model.state_dict(), path)


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
    batch_size = 10000
    learning_rate = 0.025

    # 只在主进程初始化 wandb
    if is_main_process:
        init_wandb(config={
            **config.model_dump(),
            "train": {
                "epoch": epoch,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
        })

    vocab = Vocab(VOCAB_CONFIG)
    vocab.load_vocab_from_txt()
    vocab_size = len(vocab)
    dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    cbow_dataset = CBOWDataset(
        dataset, vocab, VOCAB_CONFIG.window_size, CACHE_CONFIG.val_cbow_dataset_cache_path)

    # 添加 DistributedSampler
    train_sampler = DistributedSampler(
        cbow_dataset) if is_enable_distributed() else None
    dataloader = DataLoader(
        cbow_dataset,
        batch_size=batch_size,
        shuffle=False if train_sampler else True,
        sampler=train_sampler,
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx))

    model = CBOWModel(vocab_size, VOCAB_CONFIG.embedding_dim, vocab.pad_idx)
    model = model.to(device)
    if os.path.exists(CACHE_CONFIG.val_cbow_model_cache_path):
        state_dict = torch.load(
            CACHE_CONFIG.val_cbow_model_cache_path,
            map_location=device
        )
        model.load_state_dict(state_dict)

    origin_model = model
    if is_enable_distributed():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # type: ignore

    epoch_bar = tqdm(range(10), desc="训练", disable=local_rank != 0, position=0)
    for epoch in epoch_bar:
        if train_sampler:
            train_sampler.set_epoch(epoch)
        total_loss = 0
        batch_bar = tqdm(
            dataloader, desc=f"训练第{epoch}轮", disable=local_rank != 0, position=1)

        for i, (context_idxs, target_idx) in enumerate(batch_bar):
            context_idxs = context_idxs.to(device)
            target_idx = target_idx.to(device)
            optimizer.zero_grad()
            loss = model(context_idxs, target_idx)
            loss.backward()
            optimizer.step()

            if is_main_process:  # 只在主进程记录日志
                batch_bar.set_postfix(loss=loss.item())
                total_loss += loss.item()
                # 记录每个批次的损失
                wandb.log({"batch_loss": loss.item()},
                          step=epoch * len(dataloader) + i)

        if is_main_process:  # 只在主进程记录 epoch 级别的指标
            avg_loss = total_loss / len(dataloader)
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
