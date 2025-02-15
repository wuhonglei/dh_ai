import time
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import os

from tqdm import tqdm
import wandb

from utils.analysis import analysis_gpu_memory  # type: ignore
from utils.common import get_device
from dataset import TitleDataset, collate_fn
from transformers import BertTokenizer
from model import TitleClassifier
import atexit

def epoch_train(model, dataloader, optimizer, criterion, device):
    """ 训练一个 epoch """
    model.train()
    batch_progress = tqdm(dataloader, desc='Batch', leave=False)
    total_loss = 0
    total_correct = 0
    total_samples = 0
    support_cuda = str(device) == 'cuda'
    if support_cuda:
        scaler = GradScaler()  # 梯度缩放（防止 FP16 下溢出）

    for batch in batch_progress:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if support_cuda:
            # 前向传播使用 FP16
            with autocast(device):
                outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            # 反向传播使用 FP16，但梯度存储在 FP32 中
            scaler.scale(loss).backward()
            # 梯度缩放后更新参数（自动转回 FP32）
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
        batch_progress.set_postfix(loss=loss.item())

    loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """ 评估模型 """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    batch_progress = tqdm(dataloader, desc='Batch', leave=True)
    for batch in batch_progress:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        total_samples += labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()

    accuracy = total_correct / total_samples
    loss = total_loss / len(dataloader)

    return loss, accuracy


def train(model, epochs, train_dataloader, val_dataloader, optimizer, criterion, device, model_name: str):
    best_accuracy = 0
    epoch_progress = tqdm(range(epochs), desc='Epoch', leave=False)

    for epoch in epoch_progress:
        train_loss, train_accuracy = epoch_train(
            model, train_dataloader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(
            model, val_dataloader, criterion, device)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), f'{model_name}/best_model.pth')

        epoch_progress.set_postfix(
            train_loss=train_loss, train_accuracy=train_accuracy, val_loss=val_loss, val_accuracy=val_accuracy)
        wandb.log({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        })

    return best_accuracy


def cleanup(model, model_name):
    torch.save(model.state_dict(), f'{model_name}/model.pth')
    wandb.finish()
    # shutdown(time=10)


def main(data_dir: str, label_names: list[str], category_id_list: list[str], model_name: str, tags: list[str] = []):
    # 加载数据集
    wandb_config = {
        'project': 'shopee_title_classification',
        'config': {
            'batch_size': 128,
            'learning_rate': 0.001,
            'epochs': 5,
            'title_name': 'clean_name',
            'label_names': label_names,
            'model_name': f'./models/{model_name}',
            'num_classes': len(category_id_list),
            'bert_name': '/mnt/model/nlp/bert-base-uncased' if os.path.exists(
                '/mnt/model/nlp/bert-base-uncased') else 'bert-base-uncased',
            'device': get_device(),
        },
        'job_type': 'train',
        'tags': tags,
    }

    wandb.init(**wandb_config)
    config = wandb.config

    os.makedirs(config['model_name'], exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(config['bert_name'])

    label_encoder = LabelEncoder()
    label_encoder.fit(category_id_list)

    model_name = os.path.basename(config['model_name'])
    train_dataset = TitleDataset(data_path=f'{data_dir}/train.csv', title_name=config['title_name'],
                                 label_names=config['label_names'], tokenizer=tokenizer, cache_name=f'{model_name}/train_dataset.pkl')
    test_dataset = TitleDataset(data_path=f'{data_dir}/test.csv', title_name=config['title_name'],
                                label_names=config['label_names'], tokenizer=tokenizer, cache_name=f'{model_name}/test_dataset.pkl')
    val_dataset = TitleDataset(data_path=f'{data_dir}/val.csv', title_name=config['title_name'],
                               label_names=config['label_names'], tokenizer=tokenizer, cache_name=f'{model_name}/val_dataset.pkl')

    train_dataloader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=lambda x: collate_fn(x, label_encoder))
    test_dataloader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=lambda x: collate_fn(x, label_encoder))
    val_dataloader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=lambda x: collate_fn(x, label_encoder))

    device = config['device']
    model = TitleClassifier(num_classes=config['num_classes'],
                            bert_name=config['bert_name']).to(device)

    optimizer = optim.Adam(  # type: ignore
        model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    atexit.register(cleanup, model, config['model_name'])

    best_accuracy = train(
        model, epochs=config['epochs'], train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer, criterion=criterion, device=device, model_name=config['model_name'])

    model.load_state_dict(torch.load(
        f'{config["model_name"]}/best_model.pth', weights_only=True))
    test_loss, test_accuracy = evaluate(
        model, test_dataloader, criterion, device)
    print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}')
    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
    })
