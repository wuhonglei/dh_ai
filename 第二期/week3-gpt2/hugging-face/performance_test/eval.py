from typing import List
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from typing import TypedDict
import torch.distributed as dist
import json
import os


def is_distributed_env():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return local_rank != -1


class ExampleDict(TypedDict):
    """明确定义 Example 的 dict 包含 prompt 和 story"""
    prompt: str
    story: str


def preprocess_function(tokenizer: GPT2Tokenizer, examples: ExampleDict):
    # 合并每一条数据
    concat_examples = []
    for prompt, story in zip(examples['prompt'], examples['story']):
        concat_examples.append(f'{prompt} <|endoftext|> {story}')

    # 分别处理 prompt 和 story
    inputs = tokenizer(
        concat_examples,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )

    # 创建 labels：向右移动一位，用于下一个 token 预测
    labels = inputs['input_ids'].clone()  # type: ignore

    # 对于 prompt 部分的 token，设置为 -100（忽略不计算 loss）
    prompt_tokens = tokenizer(
        examples['prompt'],
        truncation=True,
        max_length=512
    )['input_ids']

    for i in range(len(examples['prompt'])):
        prompt_length = len(prompt_tokens[i])  # type: ignore
        labels[i][:prompt_length] = -100  # prompt 部分不计算 loss

    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels
    }


def build_dataset(prompt_path: str, story_path: str, tokenizer: GPT2Tokenizer) -> Dataset:
    prompts = Dataset.from_text(prompt_path)
    stories = Dataset.from_text(story_path)

    dataset = Dataset.from_dict({
        'prompt': prompts['text'][:10],  # type: ignore
        'story': stories['text'][:10]  # type: ignore
    })
    dataset = dataset.map(
        lambda example: preprocess_function(tokenizer, example),
        batched=True,
        batch_size=500,
    )
    return dataset


def write_to_file(text: str, file_path: str):
    with open(file_path, 'a') as f:
        f.write(text + '\n')


def main():
    is_distributed = is_distributed_env()

    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)
    print(model.config)

    test_dataset = build_dataset(
        './writingPrompts/test.wp_source',
        './writingPrompts/test.wp_target',
        tokenizer
    )

    training_args = TrainingArguments(
        report_to=[],  # 禁用所有日志上传
        output_dir='./results',
        eval_strategy='epoch',
        learning_rate=2e-5,
        local_rank=-1,
        ddp_backend="nccl" if is_distributed else None,  # 使用NCCL作为分布式后端
        per_device_eval_batch_size=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
    )

    performance = trainer.evaluate()
    if training_args.local_rank == 0:
        data = {
            'desc': 'multi process' if is_distributed else 'single process',
            'performance': performance,
        }
        # write_to_file(json.dumps(data), './eval.txt')

    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
