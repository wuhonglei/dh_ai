import torch
from transformers import AutoTokenizer, GPT2DoubleHeadsModel, GPT2Config, GPT2TokenizerFast


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


config = GPT2Config()
device = get_device()
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = GPT2DoubleHeadsModel.from_pretrained("openai-community/gpt2")

num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
embedding_layer = model.resize_token_embeddings(len(tokenizer))

choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
encoded_choices = [tokenizer.encode(o, return_tensors="pt") for o in choices]
cls_token_location = [list(tokens[0]).index(tokenizer.cls_token_id)
                      for tokens in encoded_choices]

print(cls_token_location)
input_ids = torch.stack(encoded_choices).permute(
    1, 0, 2)  # Batch size: 1, number of choices: 2
mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1
outputs = model(input_ids, mc_token_ids=mc_token_ids)
lm_logits = outputs.logits
mc_logits = outputs.mc_logits
