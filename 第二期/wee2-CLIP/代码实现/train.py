import os
import time
import torch
import pandas as pd
import torch.nn as nn
from typing import Literal
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

import wandb
from dataset import CLIPDataset, get_transforms, collate_fn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from utils.common import get_device
from tqdm import tqdm
from clip import CLIPModel, clip_loss  # type: ignore

import atexit

wandb_config = {
    'project': 'CLIP',
    'config': {
        'data': {
            'train_data_path': "datasets/train_split.csv",
            'test_data_path': "datasets/test_split.csv",
            'image_dir': "datasets/images"
        },
        'text_encoder': {
            'model_name': "distilbert-base-uncased",
            'embedding_dim': 768,
            'pretrained': False,
            'trainable': True,
            'max_length': 200
        },
        'image_encoder': {
            'model_name': "resnet50",
            'input_size': 224,
            'embedding_dim': 2048,
            'pretrained': False,
            'trainable': True
        },
        'projection_head': {
            'embedding_dim': 256,
            'dropout': 0.1
        },
        'train': {
            'batch_size': 64,
            'epochs': 3,
            'temperature': 1.0,
            'image_encoder_learning_rate': 0.001,
            'text_encoder_learning_rate': 0.001,
            'projection_head_learning_rate': 0.001,
            'loss_type': 'fixed'  # 'fixed' / 'dynamic'
        },
        'pretrained': False,
        'shutdown': False,
        'sweep': False,
        'device': get_device()
    },
    'job_type': 'train',
    'tags': ['time_cost', 'baseline']
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
        'train': {
            'parameters': {
                'loss_type': {
                    'values': ['fixed', 'dynamic']
                }
            }
        }
    }
}


def build_loader(mode: Literal['train', 'test'], config, tokenizer: DistilBertTokenizer):
    transforms = get_transforms(
        mode=mode, image_size=config['image_encoder']['input_size'])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, config['data'][f'{mode}_data_path'])
    image_dir = os.path.join(current_dir, config['data']['image_dir'])
    df = pd.read_csv(csv_path)
    image_filenames = [
        os.path.join(image_dir, image_name)
        for image_name in df['image'].tolist()
    ]
    captions = df['caption'].tolist()

    dataset = CLIPDataset(
        image_filenames=image_filenames,
        captions=captions,
        tokenizer=tokenizer,
        max_length=config['text_encoder']['max_length'],
        transforms=transforms,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True if mode == 'train' else False,
        collate_fn=collate_fn
    )
    return dataloader


def train_one_epoch(model: CLIPModel, train_loader, optimizer, loss_type, device) -> float:
    model.train()
    batch_bar = tqdm(train_loader, desc='Training', leave=False, position=1)
    total_loss = 0
    scaler = GradScaler()  # 创建梯度缩放器

    for batch in batch_bar:
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 使用autocast上下文管理器进行混合精度训练
        with autocast():
            output = model(image, input_ids, attention_mask)
            loss = clip_loss(output, loss_type)

        optimizer.zero_grad()
        # 使用scaler来处理梯度
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_bar.set_postfix(loss=loss.item())
        total_loss += loss.item()

    return total_loss / len(train_loader)


def valid_epoch(model: CLIPModel, valid_loader, loss_type, device) -> float:
    model.eval()
    total_loss = 0
    batch_bar = tqdm(valid_loader, desc='Validating', leave=False, position=2)

    with torch.no_grad(), autocast():  # 在验证时也使用混合精度
        for batch in batch_bar:
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = model(image, input_ids, attention_mask)
            loss = clip_loss(output, loss_type)
            batch_bar.set_postfix(loss=loss.item())
            total_loss += loss.item()

    return total_loss / len(valid_loader)


def cleanup(model):
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), f'./models/final_model.pth')
    print('Saved Final Model!')
    wandb.finish()


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
        temperature=config['train']['temperature']
    ).to(device)

    if config['pretrained'] and os.path.exists('./models/best_model.pth'):
        model.load_state_dict(torch.load(
            f'./models/best_model.pth', map_location=device, weights_only=True))
        print('Loaded Pretrained Model!')

    atexit.register(cleanup, model)

    optimizer_params = [{
        'params': model.image_encoder.parameters(),
        'lr': config['train']['image_encoder_learning_rate']
    }, {
        'params': model.text_encoder.parameters(),
        'lr': config['train']['text_encoder_learning_rate']
    }, {
        'params': [model.image_projection, model.text_projection],
        'lr': config['train']['projection_head_learning_rate']
    }]
    optimizer = torch.optim.AdamW(  # type: ignore
        optimizer_params, weight_decay=0.01)
    loss_type = config['train']['loss_type']
    best_loss = float('inf')

    epoch_progress = tqdm(
        range(config['train']['epochs']), desc='Epochs', leave=False, position=0)
    for epoch in epoch_progress:
        start_time = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_type, device)
        end_time = time.time()

        test_loss = valid_epoch(model, test_loader, loss_type, device)
        epoch_progress.set_postfix(
            train_loss=train_loss, test_loss=test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), f'./models/best_model.pth')

        wandb.log({
            'train_loss': train_loss,
            'test_loss': test_loss,
            'time': end_time - start_time
        })

    run.finish()


if __name__ == "__main__":
    if wandb_config['config']['sweep']:
        sweep_id = wandb.sweep(sweep_config, project=wandb_config['project'])
        wandb.agent(sweep_id, main)
    else:
        main()
