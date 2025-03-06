# 使用 t5 模型进行翻译
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("t5-large")
tokenizer = T5Tokenizer.from_pretrained("t5-large")

input_text = """
Today is sunny and cloudless
"""

# 设计提示，告诉模型我们要摘要
prompt = f"Translate the following text from English to French:\n{input_text}\nTranslation:"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=300)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("翻译结果:", translation)
