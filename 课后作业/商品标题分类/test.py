from transformers import BertModel, BertTokenizer
import torch

# 加载预训练的 BERT 模型和分词器
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 创建输入
text = "This is a sample input for BERT."
inputs = tokenizer(text, return_tensors="pt")

print(model)