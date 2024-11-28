import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import atexit

from models.vgg_crnn import CRNN
from models.util import get_transfrom_fn
from dataset import CaptchaDataset, encode_labels
from utils import get_wandb_config, EarlyStopping, save_model, correct_predictions, get_tags_from_dir
from evaluate import evaluate_model


def train(train_dir: str, test_dir: str, batch_size: int, pretrained_model_path: str, epochs: int, learning_rate: float, captcha_length: int, class_num: int, characters: str, padding_index, model_path: str, width: int, height: int, log: bool, hidden_size: int, in_channels: int, early_stopping={},):
    if log:
        wandb.init(**get_wandb_config(), job_type='train',
                   tags=get_tags_from_dir([train_dir]))

    transform = get_transfrom_fn(in_channels, height, width, 'training')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    loader_config = {'num_workers': 3, 'pin_memory': True} if is_cuda else {}
    train_dataset = CaptchaDataset(
        train_dir, captcha_length=captcha_length, characters=characters, padding_index=padding_index, transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **loader_config)

    model = CRNN(in_channels, hidden_size, class_num)
    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(
            pretrained_model_path, map_location=device))
        print('Pretrained model loaded.', pretrained_model_path)
    model.to(device)
    ctc_loss = nn.CTCLoss(blank=padding_index)  # 假设 0 为空白字符
    optimizer = optim.Adam(  # type: ignore
        model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(**early_stopping)

    def clean_up():
        print('Cleaning up...')
        save_model(model_path, model)
    atexit.register(clean_up)

    epoch_progress = tqdm(range(epochs), desc='Epoch')
    for epoch in epoch_progress:
        loss_sum = 0.0
        acc_sum = 0.0
        start_time = time.time()
        model.train()
        batch_progress = tqdm(enumerate(train_loader), total=len(
            train_loader), desc='Batch', leave=False)
        for batch_ids, (imgs, labels) in batch_progress:
            imgs = imgs.to(device)
            # Convert labels to tensor for CTC
            targets, target_lengths = encode_labels(
                labels, characters, padding_index)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            inputs = model(imgs)  # (seq_length, batch_size, n_classes)
            # Define input lengths for CTC
            input_lengths = torch.full(
                (inputs.size(1),), inputs.size(0), dtype=torch.int32).to(device)

            # Compute loss
            loss = ctc_loss(inputs.log_softmax(2), targets,
                            input_lengths, target_lengths)

            loss_sum += (loss.item() * target_lengths).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_correct, _ = correct_predictions(inputs,
                                                     labels, characters, padding_index)
            acc_sum += current_correct
            batch_progress.set_postfix(loss=f'{loss.item():.4f}')

        test_loss, test_accuracy = evaluate_model(
            test_dir, model, captcha_length, padding_index, width, height, characters, log=False, visualize=False, visualize_all=False, visualize_limit=0, in_channels=in_channels)
        train_loss, train_accuracy = loss_sum / \
            (len(train_dataset)), acc_sum / \
            (len(train_dataset))

        if log:
            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'epoch_time': int(time.time() - start_time)
            })

        early_stopping(test_loss)
        epoch_progress.set_postfix(
            test_loss=f'{test_loss:.4f}',
            train_loss=f'{train_loss:.4f}',
            test_accuracy=f'{100 * test_accuracy:.4f}%',
            train_accuracy=f'{100 * train_accuracy:.4f}%'
        )

        if early_stopping.early_stop:
            print('Early stopping in epoch:', epoch)
            break

    if log:
        wandb.finish()
