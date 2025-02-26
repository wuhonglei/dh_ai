import os
import time
import torch
import pandas as pd
import torch.nn as nn
from typing import Literal, Union, Any, cast

import wandb
from dataset import CLIPDataset, get_transforms, collate_fn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from utils.common import get_device
from tqdm import tqdm
from clip import CLIPModel, clip_loss  # type: ignore

import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
            'pretrained': True,
            'trainable': True,
            'max_length': 200
        },
        'image_encoder': {
            'model_name': "resnet50",
            'input_size': 224,
            'embedding_dim': 2048,
            'pretrained': True,
            'trainable': True
        },
        'projection_head': {
            'embedding_dim': 256,
            'dropout': 0.1
        },
        'train': {
            'batch_size': 64,
            'epochs': 10,
            'temperature': 1.0,
            'image_encoder_learning_rate': 2e-5,
            'text_encoder_learning_rate': 2e-5,
            'projection_head_learning_rate': 1e-3,
            'loss_type': 'fixed',  # 'fixed' / 'dynamic'
            'distributed': True,   # 是否使用分布式训练
            'num_workers': 4       # 数据加载的工作进程数
        },
        'pretrained': False,
        'shutdown': False,
        'sweep': False,
        'device': get_device()
    },
    'job_type': 'train',
    'tags': ['time_cost', 'multi_node_multi_gpu']
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

    if config['train']['distributed']:
        sampler = DistributedSampler(dataset, shuffle=(mode == 'train'))
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=(sampler is None and mode == 'train'),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=config['train']['num_workers'],
        pin_memory=True,
    )
    return dataloader


def train_one_epoch(model: Union[CLIPModel, DDP], train_loader, optimizer, loss_type, device) -> float:
    model.train()
    batch_bar = tqdm(train_loader, desc='Training', leave=False, position=1)
    total_loss = 0
    for batch in batch_bar:
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(
            image, input_ids, attention_mask)
        optimizer.zero_grad()
        loss = clip_loss(output, loss_type)
        loss.backward()
        optimizer.step()
        batch_bar.set_postfix(loss=loss.item())
        total_loss += loss.item()

    return total_loss / len(train_loader)


def valid_epoch(model: Union[CLIPModel, DDP], valid_loader, loss_type, device) -> float:
    model.eval()
    total_loss = 0
    batch_bar = tqdm(valid_loader, desc='Validating', leave=False, position=2)
    for batch in batch_bar:
        image = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(image, input_ids, attention_mask)
        loss = clip_loss(output, loss_type)
        batch_bar.set_postfix(loss=loss.item())
        total_loss += loss.item()

    return total_loss / len(valid_loader)


def setup_distributed(local_rank, rank, world_size):
    """
    设置分布式训练环境
    """
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 设置当前设备
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    """
    清理分布式训练环境
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def main_worker(local_rank, config):
    """
    每个GPU上运行的主要工作函数
    """
    # 获取分布式训练的环境变量
    if config['train']['distributed']:
        rank = int(os.environ.get('RANK', '0'))
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        setup_distributed(local_rank, rank, world_size)
    else:
        rank = 0
        world_size = 1

    # 只在主进程上初始化wandb
    if rank == 0:
        run = wandb.init(**wandb_config)
        config = run.config
    else:
        run = None

    device = torch.device(
        f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    tokenizer = DistilBertTokenizer.from_pretrained(
        config['text_encoder']['model_name'])

    # 创建数据加载器
    train_loader = build_loader('train', config, tokenizer)
    test_loader = build_loader('test', config, tokenizer)

    # 创建模型
    clip_model = CLIPModel(
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

    # 加载预训练模型
    if config['pretrained'] and os.path.exists('./models/best_model.pth'):
        clip_model.load_state_dict(torch.load(
            f'./models/best_model.pth', map_location=device))
        if rank == 0:
            print('Loaded Pretrained Model!')

    # 设置优化器 - 在包装为DDP前获取参数
    image_encoder_params = clip_model.image_encoder.parameters()
    text_encoder_params = clip_model.text_encoder.parameters()
    projection_params = [clip_model.image_projection,
                         clip_model.text_projection]

    # 将模型包装为DDP模型
    if config['train']['distributed']:
        model = DDP(clip_model, device_ids=[
                    local_rank], output_device=local_rank)
    else:
        model = clip_model

    optimizer_params = [
        {'params': image_encoder_params,
            'lr': config['train']['image_encoder_learning_rate']},
        {'params': text_encoder_params,
            'lr': config['train']['text_encoder_learning_rate']},
        {'params': projection_params,
            'lr': config['train']['projection_head_learning_rate']}
    ]

    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=0.01)
    loss_type = config['train']['loss_type']
    best_loss = float('inf')

    # 训练循环
    if rank == 0:
        epoch_progress = tqdm(
            range(config['train']['epochs']), desc='Epochs', leave=False, position=0)
    else:
        epoch_progress = range(config['train']['epochs'])

    if rank == 0:
        print(f'rank: {rank}, train_loader_len: {len(train_loader)}')
        print(f'rank: {rank}, test_loader_len: {len(test_loader)}')

    for epoch in epoch_progress:
        if config['train']['distributed'] and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        start_time = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_type, device)
        end_time = time.time()

        test_loss = valid_epoch(model, test_loader, loss_type, device)

        if rank == 0:
            if isinstance(epoch_progress, tqdm):
                epoch_progress.set_postfix(
                    train_loss=train_loss, test_loss=test_loss)

            if test_loss < best_loss:
                best_loss = test_loss
                os.makedirs('./models', exist_ok=True)
                # 保存模型
                if config['train']['distributed']:
                    # 保存DDP模型的module部分
                    torch.save(clip_model.state_dict(),
                               f'./models/best_model.pth')
                else:
                    torch.save(clip_model.state_dict(),
                               f'./models/best_model.pth')

            wandb.log({
                'train_loss': train_loss,
                'test_loss': test_loss,
                'time': end_time - start_time
            })

    # 清理
    if rank == 0:
        os.makedirs('./models', exist_ok=True)
        if config['train']['distributed']:
            # 保存DDP模型的module部分
            torch.save(clip_model.state_dict(), f'./models/final_model.pth')
        else:
            torch.save(clip_model.state_dict(), f'./models/final_model.pth')
        print('Saved Final Model!')
        if run:
            run.finish()

    if config['train']['distributed']:
        cleanup_distributed()


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    args = parser.parse_args()

    config = wandb_config['config']

    # 根据环境变量判断是否使用分布式训练
    if "WORLD_SIZE" in os.environ:
        config['train']['distributed'] = True

    # 调用主工作函数
    main_worker(args.local_rank, config)


if __name__ == "__main__":
    start_time = time.time()
    if wandb_config['config']['sweep']:
        sweep_id = wandb.sweep(sweep_config, project=wandb_config['project'])
        wandb.agent(sweep_id, main)
    else:
        main()
    end_time = time.time()
    print(f'Total time: {end_time - start_time} seconds')
