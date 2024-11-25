import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm

from models.crnn import CRNN, register_hook, activations
from dataset import CaptchaDataset, encode_labels
from utils import correct_predictions, load_config, get_wandb_config, wandb_image, visualize_activations


def evaluate(data_dir: str, model_path: str, captcha_length: int, class_num: int, padding_index, width: int, height: int, characters: str, hidden_size: int, log: bool, visualize: bool):
    model = CRNN(class_num, hidden_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True))
    model.to(device)
    return evaluate_model(data_dir, model, captcha_length, padding_index, width, height, characters, log, visualize)


def evaluate_model(data_dir: str, model, captcha_length: int, padding_index, width: int, height: int, characters: str, log: bool, visualize: bool):
    if log:
        wandb_config = get_wandb_config()
        wandb.init(**wandb_config, job_type='evaluate')
        # 定义表格的列
        table = wandb.Table(
            columns=["Origin_Image", "Transformed_Image", "Prediction", "Ground Truth"])

    if visualize:
        cnn_names, rnn_name = register_hook(model)
        print('cnn_names', cnn_names, rnn_name)

    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    loader_config = {'num_workers': 3, 'pin_memory': True} if is_cuda else {}
    eval_dataset = CaptchaDataset(
        data_dir, captcha_length=captcha_length, characters=characters, padding_index=padding_index, transform=transform)
    origin_eval_dataset = CaptchaDataset(
        data_dir, captcha_length=captcha_length, characters=characters, padding_index=padding_index, transform=None)
    eval_loader = DataLoader(eval_dataset, batch_size=1,
                             shuffle=False, **loader_config)

    loss_sum = 0.0
    correct = 0
    total = 0
    ctc_loss = nn.CTCLoss(blank=padding_index)  # 假设 0 为空白字符

    model.eval()
    eval_progress = tqdm(enumerate(eval_loader),
                         total=len(eval_loader), desc='Evaluating')
    for batch_index, (imgs, labels) in eval_progress:
        imgs = imgs.to(device)
        # Convert labels to tensor for CTC
        targets, target_lengths = encode_labels(
            labels, characters, padding_index)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        with torch.no_grad():
            inputs = model(imgs)  # (seq_length, batch_size, n_classes)

        # Define input lengths for CTC
        input_lengths = torch.full(
            (inputs.size(1),), inputs.size(0), dtype=torch.int32).to(device)

        # Compute loss
        loss = ctc_loss(inputs.log_softmax(2), targets,
                        input_lengths, target_lengths)
        loss_sum += (loss.item() * target_lengths).sum().item()
        total += len(labels)
        current_correct, preds = correct_predictions(inputs,
                                                     labels, characters, padding_index)
        correct += current_correct

        if log:
            wandb_image(origin_eval_dataset, eval_dataset,
                        batch_index, preds[0], labels[0], wandb, table)

        if visualize:
            visualize_activations(
                origin_eval_dataset, eval_dataset, activations, cnn_names, rnn_name, batch_index,)
            if total >= 5:
                break

    test_loss = loss_sum / total
    test_accuracy = 1.0 * correct / (total)
    if log:
        # 记录表格到 wandb
        wandb.log({"captcha_results": table, 'test_loss': test_loss,
                  'test_accuracy': test_accuracy})
        # 完成 wandb 运行
        wandb.finish()

    return test_loss, test_accuracy


if __name__ == '__main__':
    config = load_config('./config.yaml')
    model_config = config['model']
    dataset_config = config['dataset']
    testing_config = config['testing']
    evaluate_config = config['evaluate']
    class_num = len(dataset_config['characters']) + 1  # 1 表示空白字符
    test_loss, test_accuracy = evaluate(data_dir=testing_config['test_dir'],
                                        model_path=testing_config['model_path'],
                                        characters=dataset_config['characters'],
                                        hidden_size=model_config['hidden_size'],
                                        captcha_length=6,
                                        class_num=class_num,
                                        padding_index=dataset_config['padding_index'],
                                        width=model_config['width'],
                                        height=model_config['height'],
                                        log=evaluate_config['log'],
                                        visualize=evaluate_config['visualize']
                                        )

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
