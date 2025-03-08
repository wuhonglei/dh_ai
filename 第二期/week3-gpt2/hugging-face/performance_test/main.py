from typing import List
from transformers import GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer
from evaluate import load
from utils.common import get_text_token_len, get_max_token_len, write_to_file
from utils.model import create_optimized_model, setup_performance_opts
from dataset import WritingPromptsDataset, BucketSampler
from torch.utils.data import DataLoader
import torchvision
from pprint import pprint
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os

from tqdm import tqdm
import time


torchvision.disable_beta_transforms_warning()

# 启用torch的cuda优化：
setup_performance_opts()

# 设置设备和分布式训练


def setup_distributed():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    # 获取本地 rank
    local_rank = int(os.environ["LOCAL_RANK"])
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    return local_rank


# 修改设备设置部分
local_rank = setup_distributed()
device = torch.device(f"cuda:{local_rank}")
model_name = "gpt2-large"
tokenizer: PreTrainedTokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = create_optimized_model(model_name, device)
# 2. 确保模型也知道padding token的设置
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# 包装模型为 DDP 模型
model = DDP(model, device_ids=[local_rank])

# 1. 首先设置tokenizer的padding相关配置
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token


def build_loader(prompt_path, story_path, batch_size=1, shuffle=False):
    dataset = WritingPromptsDataset(
        prompt_path=prompt_path,
        story_path=story_path,
    )
    # 创建分布式采样器
    if shuffle:
        sampler = DistributedSampler(
            dataset, shuffle=True)
    else:
        sampler = BucketSampler(dataset, batch_size, lambda item: sum(
            [get_text_token_len([text])[0] for text in item.values()]
        ))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )


batch_size = 128
train_loader = build_loader(
    prompt_path="./writingPrompts/train.wp_source",
    story_path="./writingPrompts/train.wp_target",
    batch_size=batch_size,
    shuffle=True,
)

test_loader = build_loader(
    prompt_path="./writingPrompts/test.wp_source",
    story_path="./writingPrompts/test.wp_target",
    batch_size=batch_size,
    shuffle=False,
)

validation_loader = build_loader(
    prompt_path="./writingPrompts/valid.wp_source",
    story_path="./writingPrompts/valid.wp_target",
    batch_size=batch_size,
    shuffle=False,
)


def generate_text(model: PreTrainedModel | DDP, prompt_list: List[str], max_length: int = 100) -> List[str]:
    try:
        inputs = tokenizer(
            prompt_list, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,  # type: ignore
                do_sample=False,
                use_cache=True,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_list = tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        # 移除提示部分，仅保留生成内容
        return [generated.replace(prompt, "").strip() for generated, prompt in zip(generated_list, prompt_list)]
    except Exception as e:
        print(f"Error: {e}")
        return [''] * len(prompt_list)


predictions: List[str] = []
references: List[str] = []
start_time = time.time()
for batch_examples in test_loader:
    prompt: List[str] = batch_examples["prompt"]
    reference: List[str] = batch_examples["story"]
    prompt_len = get_text_token_len(prompt)
    story_len = get_text_token_len(reference)
    max_len = get_max_token_len(prompt_len, story_len)
    generated_list = generate_text(model, prompt, max_len)
    predictions.extend(generated_list)
    references.extend(reference)


# 加载评估指标
bleu = load("bleu")
rouge = load("rouge")

# 收集所有进程的预测结果
all_predictions: List[str] = []
all_references: List[str] = []
world_size = dist.get_world_size()
main_rank = 0

# 在每个进程上收集预测结果
gathered_predictions: List[List[str]] = [[] for _ in range(world_size)]
gathered_references: List[List[str]] = [[] for _ in range(world_size)]

dist.gather_object(predictions, gathered_predictions, dst=main_rank)
dist.gather_object(references, gathered_references, dst=main_rank)


# 只在主进程计算指标和写入结果
if dist.get_rank() == main_rank:
    for pred, ref in zip(gathered_predictions, gathered_references):
        all_predictions.extend(pred)
        all_references.extend(ref)

    end_time = time.time()
    # 计算 BLEU 和 ROUGE
    bleu_score = bleu.compute(
        predictions=all_predictions, references=all_references)
    rouge_score = rouge.compute(
        predictions=all_predictions, references=all_references)

    data = {
        'dataset': 'test',
        'batch_size': batch_size * world_size,  # 记录总批次大小
        "model_name": model_name,
        'total_time': end_time - start_time,
        'max_length': 'min(max(prompt_len+story_len, min_len), max_len)',
        "bleu_score": bleu_score['bleu'] if bleu_score else 0,
        "rouge_score": rouge_score
    }
    pprint(data)
    write_to_file(f"./performance.txt", data)

# 清理分布式环境
dist.destroy_process_group()
