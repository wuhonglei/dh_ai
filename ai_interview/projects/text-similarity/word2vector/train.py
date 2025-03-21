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
    vocab = Vocab(VOCAB_CONFIG)
    vocab.load_vocab_from_txt()
    dataset = NewsDatasetCsv(DATASET_CONFIG.val_csv_path)
    cbow_dataset = CBOWDataset(
        dataset, vocab, VOCAB_CONFIG.window_size, CACHE_CONFIG.val_cbow_dataset_cache_path)
    dataloader = DataLoader(
        cbow_dataset, batch_size=100, shuffle=True, collate_fn=lambda batch: collate_fn(batch, vocab.pad_idx))
    model = CBOWModel(len(vocab), VOCAB_CONFIG.embedding_dim, vocab.pad_idx)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    device = get_device()
    model.to(device)

    epoch_bar = tqdm(range(10), desc="训练")
    for epoch in epoch_bar:
        total_loss = 0
        batch_bar = tqdm(dataloader, desc=f"训练第{epoch}轮")
        for context_idxs, target_idx in batch_bar:
            context_idxs = context_idxs.to(device)
            target_idx = target_idx.to(device)
            optimizer.zero_grad()
            loss = model(context_idxs, target_idx)
            loss.backward()
            optimizer.step()
            batch_bar.set_postfix(loss=loss.item())
            total_loss += loss.item()
        epoch_bar.set_postfix(loss=total_loss / len(dataloader))
        save_model(model, CACHE_CONFIG.val_cbow_model_cache_path.replace(
            '.pth', f'_{epoch}.pth'))

    # 保存模型
    save_model(model, CACHE_CONFIG.val_cbow_model_cache_path)


if __name__ == "__main__":
    train()
