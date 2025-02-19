import torch
import torch.nn as nn
from typing import Literal

import wandb
from dataset import get_transforms, CLIPDataset
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from utils.common import get_device
from tqdm import tqdm
from clip import CLIPModel  # type: ignore

wandb_config = {
    'project': 'CLIP',
    'config': {
        'data': {
            'train_data_path': "./datasets/train_split.csv",
            'test_data_path': "./datasets/test_split.csv",
            'image_dir': "./datasets/images"
        },
        'text_encoder': {
            'model_name': "distilbert-base-uncased",
            'embedding_dim': 768,
            'pretrained': True,
            'trainable': False,
            'max_length': 200
        },
        'image_encoder': {
            'model_name': "resnet50",
            'input_size': 224,
            'embedding_dim': 2048,
            'pretrained': True,
            'trainable': False
        },
        'projection_head': {
            'embedding_dim': 256,
            'dropout': 0.1
        },
        'train': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 3,
            'temperature': 1.0,
        },
        'shutdown': False,
        'sweep': False,
        'device': get_device()
    },
    'job_type': 'train',
}

sweep_config = {
    'method': 'grid',  # 调参方法：random / grid / bayes
    'program': 'train.py',
    'metric': {
        'name': 'test_accuracy',  # 优化目标
        'goal': 'maximize'  # 最大化验证集准确率
    },
    'parameters': {
        'image_encoder': {
            'parameters': {
                'pretrained': {
                    'values': [True, False]
                },
                'trainable': {
                    'values': [True, False]
                },
            }
        },
        'text_encoder': {
            'parameters': {
                'pretrained': {
                    'values': [True, False]
                },
                'trainable': {
                    'values': [True, False]
                },
                'model_name': {
                    'values': ["distilbert-base-uncased", "bert-base-uncased"]
                }
            }
        },
        'projection_head': {
            'parameters': {
                'embedding_dim': {
                    'values': [128, 256, 512]
                },
                'dropout': {
                    'values': [0.1, 0.2, 0.3]
                }
            }
        },
    }
}


def build_loader(mode: Literal['train', 'test'], config, tokenizer: DistilBertTokenizer):
    transforms = get_transforms(
        mode=mode, image_size=config['image_encoder']['input_size'])
    dataset = CLIPDataset(
        csv_path=config['data'][f'{mode}_data_path'],
        tokenizer=tokenizer,
        max_length=config['text_encoder']['max_length'],
        transforms=transforms,
        image_dir=config['data']['image_dir']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True if mode == 'train' else False
    )
    return dataloader


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    batch_bar = tqdm(train_loader, desc='Training')
    total_loss = 0
    for batch in batch_bar:
        image = batch['image'].to(device)
        text = batch['text'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_size = image.shape[0]
        output = model(
            image, text, attention_mask)
        labels = torch.arange(batch_size).to(device)
        optimizer.zero_grad()
        image_loss = criterion(output['logits_per_image'], labels)
        text_loss = criterion(output['logits_per_text'], labels)
        loss = (image_loss + text_loss) / 2
        loss.backward()
        optimizer.step()
        scheduler.step()
        batch_bar.set_postfix(loss=loss.item())
        total_loss += loss.item()

    return total_loss / len(train_loader)


def valid_epoch(model, valid_loader, criterion, device):
    model.eval()
    batch_bar = tqdm(valid_loader, desc='Validating')
    total_loss = 0
    for batch in batch_bar:
        image = batch['image'].to(device)
        text = batch['text'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_size = image.shape[0]
        output = model(image, text, attention_mask)
        labels = torch.arange(batch_size).to(device)
        loss = criterion(output['logits_per_image'], labels)


def main():
    run = wandb.init(**wandb_config)
    config = run.config

    tokenizer = DistilBertTokenizer.from_pretrained(
        config['text_encoder']['model_name'])
    train_loader = build_loader(
        mode='train', config=config, tokenizer=tokenizer)
    test_loader = build_loader(
        mode='test', config=config, tokenizer=tokenizer)
    device = config['device']
    model = CLIPModel(
        text_model_name=config['text_encoder']['model_name'],
        image_model_name=config['image_encoder']['model_name'],
        text_pretrained=config['text_encoder']['pretrained'],
        text_trainable=config['text_encoder']['trainable'],
        image_pretrained=config['image_encoder']['pretrained'],
        image_trainable=config['image_encoder']['trainable'],
        text_embedding_dim=config['text_encoder']['embedding_dim'],
        image_embedding_dim=config['image_encoder']['embedding_dim'],
        projection_dim=config['projection_head']['embedding_dim'],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['train']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)
    epoch_progress = tqdm(range(config['train']['epochs']), desc='Epochs')
    criterion = nn.CrossEntropyLoss()
    for epoch in epoch_progress:
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     scheduler,  criterion, device)
        epoch_progress.set_postfix(train_loss=train_loss)


if __name__ == "__main__":
    if wandb_config['config']['sweep']:
        sweep_id = wandb.sweep(sweep_config, project=wandb_config['project'])
        wandb.agent(sweep_id, main)
    else:
        main()
