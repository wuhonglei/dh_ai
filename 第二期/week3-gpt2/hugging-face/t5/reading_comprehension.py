from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# 输入文本（上下文）和问题
context = """
The Eiffel Tower is a famous landmark in Paris, France. It was constructed in 1889 and stands 330 meters tall. 
The tower was built as part of the 1889 World's Fair to celebrate the 100th anniversary of the French Revolution.
"""
question = "what is the height of the Eiffel Tower?"

input_text = f"question: {question} context: {context}"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=20)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("T5 答案:", answer)
