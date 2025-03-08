from typing import List
from transformers import GPT2Tokenizer, PreTrainedTokenizer
from evaluate import load
import torchvision
from pprint import pprint
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os


from tqdm import tqdm
import time

from utils.common import get_text_token_len, get_max_token_len, write_to_file
from utils.model import create_optimized_model, generate_text
from utils.dataset import get_dataloaders


torchvision.disable_beta_transforms_warning()


def is_dist_avail_and_initialized() -> bool:
    """检查是否在分布式环境中运行"""
    # 检查是否存在 WORLD_SIZE 环境变量（torchrun 会设置此变量）
    if "WORLD_SIZE" in os.environ:
        return True
    # 检查是否已经初始化了分布式环境
    if torch.distributed.is_available() and torch.distributed.is_initialized():  # type: ignore
        return True
    return False


def setup_distributed():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, torch.device(f"cuda:{local_rank}")


def main():
    distributed = is_dist_avail_and_initialized()

    if distributed:
        local_rank, device = setup_distributed()
        is_main_process = local_rank == 0
    else:
        # 非分布式模式下，强制使用第一个GPU
        torch.cuda.set_device(0)  # 添加这行，确保使用 cuda:0
        local_rank, device = None, torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    # 模型初始化
    model_name = "gpt2-large"
    tokenizer: PreTrainedTokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = create_optimized_model(model_name, device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # 1. 首先设置tokenizer的padding相关配置
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    batch_size = 32
    k = 0.8
    train_loader, test_loader, val_loader = get_dataloaders(  # type: ignore
        ['train', 'test', 'val'], batch_size, distributed)

    predictions: List[str] = []
    references: List[str] = []
    start_time = time.time()
    test_loader = tqdm(test_loader) if is_main_process else test_loader
    for batch_examples in test_loader:
        prompt: List[str] = batch_examples["prompt"]
        reference: List[str] = batch_examples["story"]
        prompt_len = get_text_token_len(prompt)
        story_len = get_text_token_len(reference)
        max_len = get_max_token_len(prompt_len, story_len, k)
        generated_list = generate_text(
            model, tokenizer, prompt, max_len, device)
        predictions.extend(generated_list)
        references.extend(reference)

    # 加载评估指标
    bleu = load("bleu")
    rouge = load("rouge")

    # 根据是否分布式来决定如何收集结果
    if distributed:
        # 收集所有进程的预测结果
        all_predictions: List[str] = []
        all_references: List[str] = []
        world_size = dist.get_world_size()

        gathered_predictions = [[] for _ in range(world_size)]
        gathered_references = [[] for _ in range(world_size)]

        dist.gather_object(
            predictions, gathered_predictions if is_main_process else None, dst=0)
        dist.gather_object(
            references, gathered_references if is_main_process else None, dst=0)

        if is_main_process:
            for pred, ref in zip(gathered_predictions, gathered_references):
                all_predictions.extend(pred)
                all_references.extend(ref)
    else:
        all_predictions = predictions
        all_references = references

    if is_main_process:
        print('start to evaluate')
        end_time = time.time()
        # 计算 BLEU 和 ROUGE
        bleu_score = bleu.compute(
            predictions=all_predictions, references=all_references)
        rouge_score = rouge.compute(
            predictions=all_predictions, references=all_references)

        data = {
            'dataset': 'test',
            'batch_size': batch_size * (dist.get_world_size() if distributed else 1),
            "model_name": model_name,
            'total_time': int(end_time - start_time),
            'max_length': f'min(max(prompt_len+{k} * story_len, min_len), max_len)',
            "bleu_score": bleu_score['bleu'] if bleu_score else 0,
            "rouge_score": rouge_score
        }
        pprint(data)
        write_to_file(f"./performance.txt", data)

    # 如果是分布式模式，清理分布式环境
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
