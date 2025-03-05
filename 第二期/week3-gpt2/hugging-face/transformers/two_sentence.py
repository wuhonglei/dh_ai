import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput
import time

# 打印默认配置
config = GPT2Config()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 使用默认配置创建模型
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "Hello world"

inputs = tokenizer(prompt, return_tensors="pt").to(device)
print('inputs', inputs['input_ids'])
outputs = model.generate(**inputs, do_sample=False, temperature=0.9,
                         pad_token_id=tokenizer.pad_token_id, max_length=50, repetition_penalty=1.5)

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
