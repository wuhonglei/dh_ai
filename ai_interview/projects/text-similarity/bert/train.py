import os
from model import SiameseNetwork, compute_loss
from dataset import NewsDatasetCsv
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import DATASET_CONFIG, VOCAB_CONFIG, CONFIG
from vocab import Vocab
from type_definitions import WandbConfig
from utils.model import get_model_final_name
from tqdm import tqdm
import torch
from utils.common import get_device
import wandb


def collate_fn(batch: list[dict], vocab: Vocab) -> dict:
    inputs1 = vocab.batch_encoder([item["title"] for item in batch])
    inputs2 = vocab.batch_encoder([item["content"] for item in batch])
    return {
        "input_ids1": inputs1["input_ids"],
        "attention_mask1": inputs1["attention_mask"],
        "input_ids2": inputs2["input_ids"],
        "attention_mask2": inputs2["attention_mask"]
    }


def build_dataloader(csv_path: str, batch_size: int, vocab: Vocab, shuffle: bool = True) -> DataLoader:
    dataset = NewsDatasetCsv(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: collate_fn(x, vocab))


def train_one_epoch(model: SiameseNetwork, dataloader: DataLoader, optimizer: AdamW, scheduler: CosineAnnealingLR, device: torch.device) -> float:
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        inputs1 = batch["input_ids1"].to(device)
        attention_mask1 = batch["attention_mask1"].to(device)
        inputs2 = batch["input_ids2"].to(device)
        attention_mask2 = batch["attention_mask2"].to(device)
        output_1, output_2 = model.forward_pair(inputs1, attention_mask1,
                                                inputs2, attention_mask2)
        loss = compute_loss(output_1, output_2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        wandb.log({
            'batch_loss': loss.item()
        })
        total_loss += loss.item()

    return total_loss / len(dataloader)


def valid_one_epoch(model: SiameseNetwork, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0
    for batch in dataloader:
        inputs1 = batch["input_ids1"].to(device)
        attention_mask1 = batch["attention_mask1"].to(device)
        inputs2 = batch["input_ids2"].to(device)
        attention_mask2 = batch["attention_mask2"].to(device)

        with torch.no_grad():
            output_1, output_2 = model.forward_pair(inputs1, attention_mask1,
                                                    inputs2, attention_mask2)
            loss = compute_loss(output_1, output_2)
        total_loss += loss.item()
    return total_loss / len(dataloader)


project = 'bert-text-similarity'


def train(_config: dict = {}):
    wandb.init(project=project, config={
               "toml": CONFIG.model_dump(), **_config})
    config: WandbConfig = wandb.config  # type: ignore
    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    use_projection = config.use_projection
    bert_name = VOCAB_CONFIG.bert_name
    max_position_embeddings = VOCAB_CONFIG.max_length
    vocab = Vocab(bert_name, max_position_embeddings)
    device = get_device()

    train_dataloader = build_dataloader(
        DATASET_CONFIG.train_csv_path, batch_size, vocab)
    val_dataloader = build_dataloader(
        DATASET_CONFIG.val_csv_path, batch_size, vocab)

    model = SiameseNetwork(bert_name, max_position_embeddings, use_projection)
    model_final_name = get_model_final_name(config)
    print(f"model will save to {model_final_name}")
    # if os.path.exists(model_final_name):
    #     model.load_state_dict(torch.load(model_final_name))
    # else:
    #     print(f"Model {model_final_name} not found, training from scratch")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate,
                      weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, device)
        val_loss = valid_one_epoch(model, val_dataloader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_final_name.replace(
                '_final.pth', f'_best_val_loss_epoch.pth'))
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss
        })
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), model_final_name)


def main():
    use_sweep = False

    if use_sweep:
        print('use sweep')
        sweep_config = {
            'method': 'bayes',
            'metric': {
                'name': 'val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'batch_size': {
                    'values': [32]
                },
                'learning_rate': {
                    'values': [1e-5, 3e-5, 1e-4]
                },
                'weight_decay': {
                    'values': [1e-5, 1e-4]
                },
                'epochs': {
                    'values': [5, 10, 20]
                },
                'use_projection': {
                    'values': [True]
                }
            }
        }
        use_exist_sweep = True
        if use_exist_sweep:
            os.environ['WANDB_PROJECT'] = project
            sweep_id = 'tiszx0ei'
        else:
            sweep_id = wandb.sweep(sweep_config, project=project)
        wandb.agent(sweep_id, function=train, count=9)
    else:
        config = {
            'batch_size': 32,
            'learning_rate': 1e-5,
            'weight_decay': 1e-5,
            'epochs': 10,
            'use_projection': True
        }
        train(config)


if __name__ == '__main__':
    main()
