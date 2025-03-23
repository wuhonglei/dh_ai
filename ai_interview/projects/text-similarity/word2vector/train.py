from model.cbow import CBOWModel
from config import DATASET_CONFIG, VOCAB_CONFIG, CACHE_CONFIG, MILVUS_CONFIG, config, VocabConfig
from utils.common import get_device
from utils.train import is_enable_distributed, setup_distributed, cleanup_distributed, init_wandb, get_hyperparameters, get_checkpoint_path, get_checkpoint_path_final, get_train_dataset_cache_path
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
    with torch.no_grad():
        for i, (context_idxs, target_idx) in enumerate(val_loader):
            context_idxs = context_idxs.to(device)
            target_idx = target_idx.to(device)
            loss = model(context_idxs, target_idx)
            total_loss += loss.item()

    return total_loss / len(val_loader)


project = "text-similarity-word2vec_v2"


def train():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main_process = local_rank == 0
    device = get_device(is_enable_distributed(), local_rank)

    # 只在主进程初始化 wandb
    if is_main_process:
        wandb.init(project=project, config={
            "toml": config.model_dump(),
        })

    # 获取同步的超参数
    hyperparams = get_hyperparameters(is_main_process, device)

    # 使用超参数
    min_freq = hyperparams['min_freq']
    max_freq = hyperparams['max_freq']
    embedding_dim = hyperparams['embedding_dim']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    weight_decay = hyperparams['weight_decay']
    epochs = hyperparams['epochs']
    window_size = hyperparams['window_size']

    vocab = Vocab(VocabConfig(
        **{**VOCAB_CONFIG.model_dump(), 'min_freq': min_freq, 'max_freq': max_freq}))
    vocab.load_vocab_from_txt()
    vocab_size = len(vocab)

    train_csv_dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    val_csv_dataset = NewsDatasetCsv(DATASET_CONFIG.test_csv_path)

    train_dataset_cache = get_train_dataset_cache_path(
        min_freq, max_freq, window_size)
    print(f'local_rank {local_rank}, train_dataset_cache', train_dataset_cache)
    train_loader, train_sampler = build_loader(train_csv_dataset, vocab, window_size,
                                               batch_size, train_dataset_cache)
    if is_main_process:
        val_loader, val_sampler = build_loader(
            val_csv_dataset, vocab, window_size, batch_size)

    model = CBOWModel(vocab_size, embedding_dim, vocab.pad_idx)
    model = model.to(device)

    origin_model = model
    if is_enable_distributed():
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.AdamW(  # type: ignore
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_bar = tqdm(range(epochs), desc="训练",
                     disable=local_rank != 0, position=0)
    for epoch in epoch_bar:
        if train_sampler:
            train_sampler.set_epoch(epoch)
        total_loss = 0
        batch_bar = tqdm(
            train_loader, desc=f"训练第{epoch}轮", disable=local_rank != 0, position=1)
        val_loss = 0.0
        batch_len = len(train_loader)
        batch_len_10 = batch_len // 10 or 1

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
            total_loss += loss.item()

            if is_main_process:  # 只在主进程记录日志
                batch_bar.set_postfix(loss=loss.item(), val_loss=val_loss)
                # 记录每个批次的损失
                wandb.log({"batch_loss": loss.item()},
                          step=epoch * len(train_loader) + i)

                if (i + 1) % batch_len_10 == 0:
                    val_loss = evaluate(model, val_loader, device)
                    wandb.log({"val_loss": val_loss})

        scheduler.step()

        if is_enable_distributed():
            total_loss_tensor = torch.tensor(total_loss).to(device)
            # 收集所有进程的总损失
            dist.all_reduce(total_loss_tensor)
            total_loss = total_loss_tensor.item()

            # 收集所有进程的批次数
            batch_len_tensor = torch.tensor(batch_len).to(device)
            dist.all_reduce(batch_len_tensor)
            global_batch_len = batch_len_tensor.item()
        else:
            global_batch_len = batch_len

        if is_main_process:
            # 使用全局总损失除以全局总批次数
            avg_loss = total_loss / global_batch_len
            epoch_bar.set_postfix(loss=avg_loss)
            # 记录每个 epoch 的平均损失
            wandb.log({"epoch": epoch, "avg_loss": avg_loss})
            save_model(origin_model, get_checkpoint_path(hyperparams, epoch))

    # 保存最终模型并关闭 wandb（只在主进程）
    if is_main_process:
        save_model(origin_model, get_checkpoint_path_final(hyperparams))
        wandb.finish()


def main():
    if is_enable_distributed():
        local_rank = int(os.environ['LOCAL_RANK'])
        setup_distributed(local_rank)
    else:
        local_rank = 0
    is_main_process = local_rank == 0

    if is_main_process:
        sweep_config = {
            'method': 'bayes',
            'metric': {'name': 'val_loss', 'goal': 'minimize'},
            'parameters': {
                'min_freq': {'values': [350, 500]},
                'max_freq': {'values': [80501, 105190]},
                'embedding_dim': {'values': [100, 200, 300]},
                'batch_size': {'values': [256, 512, 1024, 12800]},
                'learning_rate': {'values': [1e-3, 3e-3, 1e-2]},
                'weight_decay': {'values': [1e-4, 1e-3]},
                'epochs': {'values': [5, 10]},
                'window_size': {'values': [2, 5, 8]},
            }
        }
        use_exist_sweep = True
        if use_exist_sweep:
            os.environ['WANDB_PROJECT'] = project
            sweep_id = '47gmlelw'
        else:
            sweep_id = wandb.sweep(sweep_config, project=project)
        wandb.agent(sweep_id, function=train, count=30)
    else:
        train()

    # 清理分布式进程组
    cleanup_distributed()


if __name__ == "__main__":
    main()
