from model import SiameseNetwork, compute_loss
from config import DATASET_CONFIG, VOCAB_CONFIG, CONFIG, VocabConfig, PROJECT, CACHE_CONFIG
from utils.common import get_device
from utils.train import get_checkpoint_path_final, get_best_checkpoint_path
from vocab import Vocab
from dataset import NewsDatasetCsv
from torch.utils.data import DataLoader
import torch.optim as optim
from type_definitions import WandbConfig, NewsItem
from tqdm import tqdm
import torch
import os
import wandb
from torch.nn.utils.rnn import pad_sequence


def save_model(model: SiameseNetwork, path: str):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    torch.save(model.state_dict(), path)


def collate_fn(batch: list[NewsItem], vocab: Vocab, max_title_length: int, max_content_length: int):
    title_token_indices: list[torch.Tensor] = []
    content_token_indices: list[torch.Tensor] = []

    for item in batch:
        # 遍历每个样本
        title_tokens = vocab.tokenize(item["title"])
        content_tokens = vocab.tokenize(item["content"])
        title_indices = vocab.batch_encoder(title_tokens)
        content_indices = vocab.batch_encoder(content_tokens)

        title_token_indices.append(torch.LongTensor(
            title_indices[:max_title_length]))
        content_token_indices.append(torch.LongTensor(
            content_indices[:max_content_length]))

    clipped_title_token_indices = pad_sequence(
        title_token_indices, padding_value=vocab.pad_idx, batch_first=True)
    clipped_content_token_indices = pad_sequence(
        content_token_indices, padding_value=vocab.pad_idx, batch_first=True)

    return {
        "input_ids_title": clipped_title_token_indices,
        "input_ids_content": clipped_content_token_indices
    }


def build_dataloader(csv_path: str, batch_size: int, vocab: Vocab, max_title_length: int, max_content_length: int, shuffle: bool = True) -> DataLoader:
    dataset = NewsDatasetCsv(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: collate_fn(x, vocab, max_title_length, max_content_length))


def train_one_epoch(model: SiameseNetwork, dataloader: DataLoader, optimizer: optim.AdamW, temperature: float, device: torch.device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids_title = batch["input_ids_title"].to(device)
        input_ids_content = batch["input_ids_content"].to(device)
        optimizer.zero_grad()
        output_1, output_2 = model.forward_pair(
            input_ids_title, input_ids_content)
        loss = compute_loss(output_1, output_2, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)  # 防止梯度爆炸
        optimizer.step()
        wandb.log({"batch_loss": loss.item()})
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_one_epoch(model: SiameseNetwork, val_loader: DataLoader, temperature: float, device: torch.device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证"):
            input_ids_title = batch["input_ids_title"].to(device)
            input_ids_content = batch["input_ids_content"].to(device)
            output_1, output_2 = model.forward_pair(
                input_ids_title, input_ids_content)
            loss = compute_loss(output_1, output_2, temperature)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train(_config: dict = {}):
    device = get_device()
    wandb.init(project=PROJECT, config={
        "toml": CONFIG.model_dump(),
        **_config
    })

    config: WandbConfig = wandb.config  # type: ignore

    # 使用超参数
    min_freq = config.min_freq
    max_freq = config.max_freq
    embedding_dim = config.embedding_dim
    projection_dim = config.projection_dim
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    epochs = config.epochs
    temperature = config.temperature
    max_title_length = config.max_title_length
    max_content_length = config.max_content_length
    use_pretrained_model = config.use_pretrained_model
    pre_trained_model_path = CACHE_CONFIG.pre_trained_model_path

    vocab = Vocab(VocabConfig(
        **{**VOCAB_CONFIG.model_dump(), 'min_freq': min_freq, 'max_freq': max_freq}))
    vocab.load_vocab_from_txt()
    vocab_size = len(vocab)

    train_dataloader = build_dataloader(
        DATASET_CONFIG.train_csv_path, batch_size, vocab, max_title_length, max_content_length)
    val_dataloader = build_dataloader(
        DATASET_CONFIG.val_csv_path, batch_size, vocab, max_title_length, max_content_length)

    model = SiameseNetwork(vocab_size, embedding_dim,
                           projection_dim, vocab.pad_idx)
    if use_pretrained_model:
        print(f"Loading pretrained model from {pre_trained_model_path}")
        model.load_pretrained_embedding_model(pre_trained_model_path)
    model = model.to(device)

    optimizer = optim.AdamW(  # type: ignore
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_bar = tqdm(range(epochs), desc="训练")
    best_val_loss = float('inf')
    for epoch in epoch_bar:
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, temperature, device)
        val_loss = evaluate_one_epoch(
            model, val_dataloader, temperature, device)
        scheduler.step()
        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)
        # 记录每个 epoch 的平均损失
        wandb.log(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, get_best_checkpoint_path(config))

    # 保存最终模型并关闭 wandb
    final_model_path = get_checkpoint_path_final(config)
    save_model(model, final_model_path)
    wandb.summary['final_model_path'] = final_model_path
    wandb.finish()


def main():
    use_sweep = False
    if not use_sweep:
        config = {
            'min_freq': VOCAB_CONFIG.min_freq,
            'max_freq': VOCAB_CONFIG.max_freq,
            'embedding_dim': VOCAB_CONFIG.embedding_dim,
            'projection_dim': VOCAB_CONFIG.projection_dim,
            'batch_size': VOCAB_CONFIG.batch_size,
            'learning_rate': VOCAB_CONFIG.learning_rate,
            'weight_decay': VOCAB_CONFIG.weight_decay,
            'epochs': VOCAB_CONFIG.epochs,
            'temperature': VOCAB_CONFIG.temperature,
            'max_title_length': VOCAB_CONFIG.max_title_length,
            'max_content_length': VOCAB_CONFIG.max_content_length,
            'use_pretrained_model': VOCAB_CONFIG.use_pretrained_model,
        }
        train(config)
        return

    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'min_freq': {'values': [350]},
            'max_freq': {'values': [15000000]},
            'embedding_dim': {'values': [200]},
            'batch_size': {'values': [512]},
            'learning_rate': {'values': [3e-3]},
            'weight_decay': {'values': [1e-4]},
            'epochs': {'values': [10]},
            'temperature': {'values': [0.07]},
            'max_title_length': {'values': [16]},
            'max_content_length': {'values': [512]},
            'use_pretrained_model': {'values': [True]},
        }
    }
    use_exist_sweep = False
    if use_exist_sweep:
        os.environ['WANDB_PROJECT'] = PROJECT
        sweep_id = 't4t1cue8'
    else:
        sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    wandb.agent(sweep_id, function=train, count=1)  # 不指明 count 会无限运行


if __name__ == "__main__":
    main()
