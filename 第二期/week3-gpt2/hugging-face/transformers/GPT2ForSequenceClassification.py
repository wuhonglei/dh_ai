import torch
from transformers import AutoTokenizer, GPT2ForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialogRPT-updown")
model = GPT2ForSequenceClassification.from_pretrained(
    "microsoft/DialogRPT-updown")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = GPT2ForSequenceClassification.from_pretrained(
    "microsoft/DialogRPT-updown", num_labels=num_labels)

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss
