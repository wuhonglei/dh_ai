import torch
from dataset import TextDataset, collate_fn
from model import GPT2LMHeadModel
from config import GPT2Config
from encoder import Encoder
from utils import get_device, load_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
from sample import sample_sequence
import jieba


def train():
    device = get_device()
    encoder = Encoder("data/vocab.txt")
    pad_token_id = encoder.vocab.index('<pad>')
    dataset = TextDataset("data/train.txt")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                            collate_fn=lambda batch: collate_fn(batch, encoder, pad_token_id=pad_token_id))
    config = GPT2Config(**{
        'vocab_size_or_config_json_file': len(encoder),
        'n_positions': 1024,
        'n_ctx': 1024,
        'n_embd': 32,
        'n_layer': 2,
        'n_head': 2,
        'layer_norm_epsilon': 1e-5,
    })
    model = GPT2LMHeadModel(config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_progress = tqdm(range(150), desc="Epoch")

    prompt = input('请输入prompt:')
    for epoch in epoch_progress:
        epoch_progress.write(''.join(encoder.decode(sample_sequence(
            model, 25, context=encoder.encode(jieba.lcut(prompt)), device=device, sample=False, top_k=10)[0].tolist())))
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss = model(input_ids, lm_labels=labels,
                         ignore_index=pad_token_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_progress.set_postfix(loss=loss.item())


if __name__ == "__main__":
    train()
