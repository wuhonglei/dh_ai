import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from evaluate import load
from tqdm import tqdm
from utils.common import get_device, get_text_token_len, get_max_token_len, write_to_file
from dataset import WritingPromptsDataset
import torchvision
from pprint import pprint

torchvision.disable_beta_transforms_warning()

model_name = "gpt2-large"  # 可替换为 "gpt2-medium", "gpt2-large" 等
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 设置设备
device = get_device()
model.to(device)  # type: ignore

train_dataset = WritingPromptsDataset(
    prompt_path="./writingPrompts/train.wp_source",
    story_path="./writingPrompts/test.wp_target",
)

test_dataset = WritingPromptsDataset(
    prompt_path="./writingPrompts/test.wp_source",
    story_path="./writingPrompts/test.wp_target",
)

validation_dataset = WritingPromptsDataset(
    prompt_path="./writingPrompts/valid.wp_source",
    story_path="./writingPrompts/valid.wp_target",
)


def generate_text(model, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt",
                       max_length=1024, truncation=True).to(device)
    try:
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=3,  # 避免重复
            # top_p=0.95,  # 使用 nucleus sampling
            # temperature=0.7,  # 控制随机性
            pad_token_id=tokenizer.eos_token_id  # 避免警告
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除提示部分，仅保留生成内容
        return generated.replace(prompt, "").strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""


# 加载评估指标
bleu = load("bleu")
rouge = load("rouge")

# 生成和收集结果
predictions = []
references = []

progress = tqdm(test_dataset, desc="Generating Text")  # type: ignore
for example in progress:
    prompt = example["prompt"]
    reference = example["story"]
    prompt_len = get_text_token_len(prompt)
    story_len = get_text_token_len(reference)
    max_len = get_max_token_len(prompt_len, story_len)
    generated = generate_text(model, prompt, max_len)
    predictions.append(generated)
    references.append(reference)
    break


# 计算 BLEU 和 ROUGE
bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)

data = {
    "model_name": model_name,
    'dataset': 'test',
    'max_length': 'min(max(prompt_len+story_len, min_len), max_len)',
    "bleu_score": bleu_score['bleu'],  # type: ignore
    "rouge_score": rouge_score
}
pprint(data)
write_to_file(f"./performance.txt", data)
