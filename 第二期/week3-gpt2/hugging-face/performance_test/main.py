from typing import List
from transformers import GPT2Tokenizer, GPT2LMHeadModel, PreTrainedModel, PreTrainedTokenizer
from evaluate import load
from utils.common import get_device, get_text_token_len, get_max_token_len, write_to_file
from utils.model import create_optimized_model
from dataset import WritingPromptsDataset, BucketSampler
from torch.utils.data import DataLoader
import torchvision
from pprint import pprint
import torch

from tqdm import tqdm
import time


torchvision.disable_beta_transforms_warning()

# 启用torch的cuda优化：
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "gpt2-medium"  # 可替换为 "gpt2-medium", "gpt2-large" 等
tokenizer: PreTrainedTokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = create_optimized_model(model_name, device)

# 1. 首先设置tokenizer的padding相关配置
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# 2. 确保模型也知道padding token的设置
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id


def build_loader(prompt_path, story_path, batch_size=1, shuffle=False):
    dataset = WritingPromptsDataset(
        prompt_path=prompt_path,
        story_path=story_path,
    )
    sampler = BucketSampler(dataset, batch_size, lambda item: sum(
        [
            get_text_token_len([text])[0] for text in item.values()
        ]
    ))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=None if shuffle else sampler)


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


def generate_text(model: PreTrainedModel, prompt_list: List[str], max_length: int = 100) -> List[str]:
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


# 加载评估指标
bleu = load("bleu")
rouge = load("rouge")

# 生成和收集结果
predictions = []
references = []

progress = tqdm(test_loader, desc="Generating Text")  # type: ignore
start_time = time.time()
for batch_examples in progress:
    prompt = batch_examples["prompt"]
    reference = batch_examples["story"]
    prompt_len = get_text_token_len(prompt)
    story_len = get_text_token_len(reference)
    max_len = get_max_token_len(prompt_len, story_len)
    generated_list = generate_text(model, prompt, max_len)
    predictions.extend(generated_list)
    references.extend(reference)

end_time = time.time()
# 计算 BLEU 和 ROUGE
bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)

data = {
    'dataset': 'test',
    'batch_size': batch_size,
    "model_name": model_name,
    'total_time': end_time - start_time,
    'max_length': 'min(max(prompt_len+story_len, min_len), max_len)',
    "bleu_score": bleu_score['bleu'],  # type: ignore
    "rouge_score": rouge_score
}
pprint(data)
write_to_file(f"./performance.txt", data)
