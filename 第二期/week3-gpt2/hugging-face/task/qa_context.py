import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载预训练的 GPT-2 模型和分词器
model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # type: ignore

# 输入上下文和问题
context = """
The Eiffel Tower is a famous landmark in Paris, France. It was constructed in 1889 and stands 330 meters tall.
"""
question = "What is the height of the Eiffel Tower?"

# 设计提示，告诉模型根据上下文回答问题
prompt = f"Based on the context, answer the question concisely:\nContext: {context}\nQuestion: {question}\nAnswer:"

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt",
                   max_length=1024, truncation=True)
inputs = {key: value.to(device) for key, value in inputs.items()}

# 生成答案
outputs = model.generate(
    inputs["input_ids"],
    max_length=100,  # 控制答案长度
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_p=0.95,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

# 解码生成的答案
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = answer.split("Answer:")[1].strip() if "Answer:" in answer else answer
print("生成的答案:", answer)
