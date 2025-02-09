from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import os

from tqdm import tqdm
import wandb

from dataset import TitleDataset, collate_fn
from utils.category import load_category_list
from transformers import BertTokenizer
from model import TitleClassifier
import atexit
from shutdown import shutdown

# 加载数据集
category_list = load_category_list('./json/mtsku_category_tree.json')
wandb_config = {
    'project': 'shopee_title_classification',
    'config': {
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 10,
        'num_classes': len(category_list),
        'bert_name': '/mnt/model/nlp/bert-base-uncased' if os.path.exists(
            '/mnt/model/nlp/bert-base-uncased') else 'bert-base-uncased',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    },
    'job_type': 'train',
}

wandb.init(**wandb_config)
config = wandb.config

tokenizer = BertTokenizer.from_pretrained(config['bert_name'])


label_encoder = LabelEncoder()
label_encoder.fit([str(item['id']) for item in category_list])

train_dataset = TitleDataset(data_path='./csv/train.csv')
test_dataset = TitleDataset(data_path='./csv/test.csv')
val_dataset = TitleDataset(data_path='./csv/val.csv')


def collate_fn_wrapper(batch):
    return collate_fn(batch, label_encoder, tokenizer)


train_dataloader = DataLoader(
    train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn_wrapper)
test_dataloader = DataLoader(
    test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn_wrapper)
val_dataloader = DataLoader(
    val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn_wrapper)

device = torch.device(config['device'])
model = TitleClassifier(num_classes=config['num_classes'],
                        bert_name=config['bert_name']).to(device)

optimizer = optim.Adam(  # type: ignore
    model.parameters(), lr=config['learning_rate'])
criterion = nn.CrossEntropyLoss()


def epoch_train(model, dataloader):
    """ 训练一个 epoch """
    model.train()
    batch_progress = tqdm(dataloader, desc='Batch', leave=False)
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in batch_progress:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
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


def evaluate(model, dataloader):
    """ 评估模型 """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    batch_progress = tqdm(dataloader, desc='Batch', leave=False)
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


def train(model, epochs, train_dataloader, val_dataloader):
    best_accuracy = 0
    epoch_progress = tqdm(range(epochs), desc='Epoch', leave=False)
    for epoch in epoch_progress:
        train_loss, train_accuracy = epoch_train(model, train_dataloader)
        val_loss, val_accuracy = evaluate(model, val_dataloader)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

        epoch_progress.set_postfix(
            train_loss=train_loss, train_accuracy=train_accuracy, val_loss=val_loss, val_accuracy=val_accuracy)
        wandb.log({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        })

    return best_accuracy


def cleanup():
    torch.save(model.state_dict(), 'model.pth')
    wandb.finish()
    # shutdown(time=10)


if __name__ == '__main__':
    atexit.register(cleanup)

    best_accuracy = train(
        model, epochs=config['epochs'], train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    print(f'Best Accuracy: {best_accuracy:.4f}')

    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_accuracy = evaluate(model, test_dataloader)
    print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}')
    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
    })
