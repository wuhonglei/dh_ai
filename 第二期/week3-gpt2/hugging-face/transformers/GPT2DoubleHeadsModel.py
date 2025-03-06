import torch
from transformers import AutoTokenizer, GPT2DoubleHeadsModel

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2DoubleHeadsModel.from_pretrained("openai-community/gpt2")

# Add a [CLS] to the vocabulary (we should train it also!)
num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
# Update the model embeddings with the new vocabulary size
embedding_layer = model.resize_token_embeddings(len(tokenizer))

choices = ["The capital of France is Paris [CLS]",
           "The capital of France is London [CLS]"]
encoded_choices = [tokenizer.encode(s) for s in choices]
cls_token_location = [tokens.index(tokenizer.cls_token_id)
                      for tokens in encoded_choices]

input_ids = torch.tensor(encoded_choices).unsqueeze(
    0)  # Batch size: 1, number of choices: 2
mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

outputs = model(input_ids, mc_token_ids=mc_token_ids)
lm_logits = outputs.logits
mc_logits = outputs.mc_logits

print("多选分类 logits:", mc_logits)
predicted_choice = torch.argmax(mc_logits, dim=1)
print("预测的选项索引:", predicted_choice.item())  # 应该输出 0，表示 "Paris"
