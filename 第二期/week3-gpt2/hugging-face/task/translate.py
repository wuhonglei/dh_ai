from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载预训练的 GPT-2 模型和分词器
model_name = "gpt2-large"  # 可以使用 "gpt2-medium" 或 "gpt2-large" 以获得更好的效果
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 设置pad_token为eos_token
tokenizer.pad_token = tokenizer.eos_token


# 输入文本（需要摘要的原文）
input_text = """
Today is sunny and cloudless
"""

# 设计提示，告诉模型我们要摘要
prompt = f"Translate the following text from English to French:\n{input_text}\nTranslation:"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt",
                   max_length=1024, truncation=True)

# 生成摘要
outputs = model.generate(
    **inputs,
    do_sample=True,
    max_length=300,  # 控制摘要长度
    num_return_sequences=1,
    no_repeat_ngram_size=2,  # 避免重复
    top_p=0.95,  # 使用 nucleus sampling
    temperature=0.7,  # 控制生成随机性
    pad_token_id=tokenizer.eos_token_id  # 设置填充 token，避免警告
)

# 解码生成的摘要
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 提取 "Translation:" 之后的文本
translation = translation.split("Translation:")[1].strip(
) if "Translation:" in translation else translation
print("生成的翻译:", translation)
