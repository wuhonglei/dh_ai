from model.cbow import CBOWModel
from config import DATASET_CONFIG, VOCAB_CONFIG, CACHE_CONFIG
from utils.common import load_pickle_file, save_pickle_file, get_device
from cbow_dataset import CBOWDataset
from vocab import Vocab
from dataset import NewsDatasetCsv
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import os


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
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    vocab = Vocab(VOCAB_CONFIG)
    vocab.load_vocab_from_txt()
    vocab_size = len(vocab)
    dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    cbow_dataset = CBOWDataset(
        dataset, vocab, VOCAB_CONFIG.window_size, CACHE_CONFIG.val_cbow_dataset_cache_path)

    # 添加 DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        cbow_dataset)
    dataloader = DataLoader(
        cbow_dataset,
        batch_size=100,
        shuffle=False,  # 使用 DistributedSampler 时需要设置为 False
        sampler=train_sampler,
        collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx))

    model = CBOWModel(vocab_size, VOCAB_CONFIG.embedding_dim, vocab.pad_idx)
    # 将模型转换为 DDP 模型
    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank])

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epoch_bar = tqdm(range(10), desc="训练", disable=local_rank != 0)
    for epoch in epoch_bar:
        train_sampler.set_epoch(epoch)  # 确保每个 epoch 的数据打乱不同
        total_loss = 0
        batch_bar = tqdm(
            dataloader, desc=f"训练第{epoch}轮", disable=local_rank != 0)

        for context_idxs, target_idx in batch_bar:
            context_idxs = context_idxs.cuda(local_rank)
            target_idx = target_idx.cuda(local_rank)
            optimizer.zero_grad()
            loss = model(context_idxs, target_idx)
            loss.backward()
            optimizer.step()

            if local_rank == 0:  # 只在主进程显示进度
                batch_bar.set_postfix(loss=loss.item())
                total_loss += loss.item()

        if local_rank == 0:  # 只在主进程保存模型
            epoch_bar.set_postfix(loss=total_loss / len(dataloader))
            save_model(model.module, CACHE_CONFIG.val_cbow_model_cache_path.replace(
                '.pth', f'_{epoch}.pth'))

    # 保存最终模型（只在主进程）
    if local_rank == 0:
        save_model(model.module, CACHE_CONFIG.val_cbow_model_cache_path)

    # 清理分布式进程组
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
