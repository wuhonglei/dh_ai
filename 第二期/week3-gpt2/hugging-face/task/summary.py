from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载预训练的 GPT-2 模型和分词器
model_name = "gpt2-large"  # 可以使用 "gpt2-medium" 或 "gpt2-large" 以获得更好的效果
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 设置pad_token为eos_token
tokenizer.pad_token = tokenizer.eos_token


# 输入文本（需要摘要的原文）
input_text = """
The quick brown fox jumps over the lazy dog. This is a classic example used to test typing skills. 
The fox is known for its speed and agility, while the dog is often seen as more relaxed. 
This sentence has been widely used in various contexts, including typography and programming.
"""

# 设计提示，告诉模型我们要摘要
prompt = f"Summarize the following text:\n{input_text}\nSummary:"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt",
                   max_length=1024, truncation=True)

# 生成摘要
outputs = model.generate(
    **inputs,
    do_sample=True,
    max_length=100,  # 控制摘要长度
    num_return_sequences=1,
    no_repeat_ngram_size=2,  # 避免重复
    top_p=0.95,  # 使用 nucleus sampling
    temperature=0.7,  # 控制生成随机性
)

# 解码生成的摘要
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("生成的摘要:", summary)
