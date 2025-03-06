from transformers import AutoTokenizer, GPT2ForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2ForQuestionAnswering.from_pretrained("openai-community/gpt2")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0,
                                         answer_start_index: answer_end_index + 1]

# target is "nice puppet"
target_start_index = torch.tensor([14])
target_end_index = torch.tensor([15])

outputs = model(**inputs, start_positions=target_start_index,
                end_positions=target_end_index)
loss = outputs.loss
