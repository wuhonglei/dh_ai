import os
import time
import torch
import pandas as pd
import torch.nn as nn
from typing import Literal

import wandb
from dataset import CLIPDataset, get_transforms, collate_fn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from utils.common import get_device
from tqdm import tqdm
from clip import CLIPModel, clip_loss  # type: ignore

import atexit
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
    'tags': ['time_cost', 'single_node_2_gpu']
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


def train_one_epoch(model: CLIPModel, train_loader, optimizer, loss_type, device) -> float:
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


def valid_epoch(model: CLIPModel, valid_loader, loss_type, device) -> float:
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


def setup_distributed(rank, world_size):
    """
    设置分布式训练环境
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 设置当前设备
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """
    清理分布式训练环境
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def main_worker(rank, world_size, config):
    """
    每个GPU上运行的主要工作函数
    """
    if config['train']['distributed']:
        setup_distributed(rank, world_size)

    # 只在主进程上初始化wandb
    if rank == 0:
        run = wandb.init(**wandb_config)
        config = run.config
    else:
        run = None

    device = torch.device(
        f'cuda:{rank}' if 'cuda' in config['device'] else config['device'])

    tokenizer = DistilBertTokenizer.from_pretrained(
        config['text_encoder']['model_name'])

    # 创建数据加载器
    train_loader = build_loader('train', config, tokenizer)
    test_loader = build_loader('test', config, tokenizer)

    # 创建模型
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

    # 加载预训练模型
    if config['pretrained'] and os.path.exists('./models/best_model.pth'):
        model.load_state_dict(torch.load(
            f'./models/best_model.pth', map_location=device, weights_only=True))
        if rank == 0:
            print('Loaded Pretrained Model!')

    # 将模型包装为DDP模型
    if config['train']['distributed']:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # 设置优化器
    optimizer_params = [{
        'params': model.module.image_encoder.parameters() if config['train']['distributed'] else model.image_encoder.parameters(),
        'lr': config['train']['image_encoder_learning_rate']
    }, {
        'params': model.module.text_encoder.parameters() if config['train']['distributed'] else model.text_encoder.parameters(),
        'lr': config['train']['text_encoder_learning_rate']
    }, {
        'params': [model.module.image_projection, model.module.text_projection] if config['train']['distributed']
        else [model.image_projection, model.text_projection],
        'lr': config['train']['projection_head_learning_rate']
    }]

    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=0.01)
    loss_type = config['train']['loss_type']
    best_loss = float('inf')

    # 训练循环
    if rank == 0:
        epoch_progress = tqdm(
            range(config['train']['epochs']), desc='Epochs', leave=False, position=0)
    else:
        epoch_progress = range(config['train']['epochs'])

    print(f'rank: {rank}, train_loader_len: {len(train_loader)}')
    print(f'rank: {rank}, test_loader_len: {len(test_loader)}')

    for epoch in epoch_progress:
        if config['train']['distributed']:
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
                # 保存非DDP模型
                if config['train']['distributed']:
                    torch.save(model.module.state_dict(),
                               f'./models/best_model.pth')
                else:
                    torch.save(model.state_dict(), f'./models/best_model.pth')

            wandb.log({
                'train_loss': train_loss,
                'test_loss': test_loss,
                'time': end_time - start_time
            })

    # 清理
    if rank == 0:
        os.makedirs('./models', exist_ok=True)
        if config['train']['distributed']:
            torch.save(model.module.state_dict(), f'./models/final_model.pth')
        else:
            torch.save(model.state_dict(), f'./models/final_model.pth')
        print('Saved Final Model!')
        if run:
            run.finish()

    if config['train']['distributed']:
        cleanup_distributed()


def main():
    config = wandb_config['config']

    if config['train']['distributed']:
        # 获取可用的GPU数量
        world_size = torch.cuda.device_count()
        if world_size > 1:
            # 使用torch.multiprocessing启动多个进程
            import torch.multiprocessing as mp
            mp.spawn(main_worker, args=(world_size, config),
                     nprocs=world_size, join=True)
        else:
            print("警告：分布式训练需要多个GPU，但只找到一个。将使用单GPU训练。")
            config['train']['distributed'] = False
            main_worker(0, 1, config)
    else:
        # 单GPU训练
        main_worker(0, 1, config)


if __name__ == "__main__":
    start_time = time.time()
    if wandb_config['config']['sweep']:
        sweep_id = wandb.sweep(sweep_config, project=wandb_config['project'])
        wandb.agent(sweep_id, main)
    else:
        main()
    end_time = time.time()
    print(f'Total time: {end_time - start_time} seconds')
