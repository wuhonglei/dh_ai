""" 开放性问答 """

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载预训练的 GPT-2 模型和分词器
model_name = "gpt2-xl"  # 可替换为 "gpt2-medium" 或 "gpt2-large" 以提升效果
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # type: ignore

# 输入问题
question = "What is the capital of France?"

# 设计提示，告诉模型我们要回答问题
prompt = f"Answer the following question concisely:\nQuestion: {question}\nAnswer:"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt",
                   max_length=1024, truncation=True)
inputs = {key: value.to(device) for key, value in inputs.items()}  # 移动到设备

# 生成答案
outputs = model.generate(
    inputs["input_ids"],
    do_sample=False,
    max_length=30,  # 控制答案长度
    num_return_sequences=1,
    no_repeat_ngram_size=2,  # 避免重复
    top_p=0.95,  # 使用 nucleus sampling
    temperature=0.7,  # 控制生成随机性
    pad_token_id=tokenizer.eos_token_id  # 设置填充 token，避免警告
)

# 解码生成的答案
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
# 提取 "Answer:" 之后的文本
answer = answer.split("Answer:")[1].strip() if "Answer:" in answer else answer
print("生成的答案:", answer)
