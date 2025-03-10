from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from accelerate import DistributedDataParallelKwargs
import torch.distributed as dist
import os

import warnings
warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*")


def preprocess_function(tokenizer: GPT2Tokenizer, examples: dict):
    # 组合文本
    combined_texts = [
        f"{prompt}<|endoftext|>{story}<|endoftext|>"
        for prompt, story in zip(examples['prompt'], examples['story'])
    ]

    # 使用 tokenizer 处理文本
    encodings = tokenizer(
        combined_texts,
        truncation=True,
        max_length=900,
        padding="max_length",
        return_tensors="pt"
    )

    # 添加 labels
    encodings['labels'] = encodings['input_ids'].clone()  # type: ignore

    return encodings


def build_dataset(prompt_path: str, story_path: str, tokenizer: GPT2Tokenizer) -> Dataset:
    prompts = Dataset.from_text(prompt_path)
    stories = Dataset.from_text(story_path)
    raw_dataset = Dataset.from_dict({
        'prompt': prompts['text'][:100],  # type: ignore
        'story': stories['text'][:100]  # type: ignore
    })

    processed_dataset = raw_dataset.map(
        lambda x: preprocess_function(tokenizer, x),
        batched=True,
        batch_size=10,
    )

    # 设置数据格式
    processed_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )

    return processed_dataset


def main():
    # 1. 加载预训练模型和分词器
    model_name = "gpt2-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(
        model_name)
    model.config.use_cache = False  # 添加这行，避免与 gradient checkpointing 冲突

    # 2. 设置特殊token
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    # 确保模型知道 pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = build_dataset(
        "writingPrompts/train.wp_source", "writingPrompts/train.wp_target", tokenizer)
    valid_dataset = build_dataset(
        "writingPrompts/valid.wp_source", "writingPrompts/valid.wp_target", tokenizer)

    # 5. 定义训练参数
    training_args = TrainingArguments(
        output_dir="./gpt2-writing-prompts",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=2e-5,
        eval_strategy="no",
        eval_steps=1000,

        # 分布式训练相关参数
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        fp16=True,  # 使用混合精度训练
        gradient_checkpointing=True,  # 启用梯度检查点以节省显存

        # DDP 设置
        ddp_backend="nccl",  # GPU 训练推荐使用 nccl
        ddp_find_unused_parameters=False,

        # 数据加载设置
        dataloader_num_workers=4,
    )

    # 6. 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=valid_dataset,
    )

    # 7. 开始训练
    trainer.train()

    # 8. 只在主进程中保存模型
    if training_args.local_rank == 0:
        print("正在保存模型...")
        model.save_pretrained("./gpt2-writing-prompts-final")
        tokenizer.save_pretrained("./gpt2-writing-prompts-final")
        print("模型保存完成")

    # 9. 清理分布式训练资源
    if training_args.local_rank != -1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
