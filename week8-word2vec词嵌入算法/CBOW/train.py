import os
import jieba
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TextReader
from model import CBow


def read_dir(directory: str) -> list[str]:
    tokens = []
    frequency = {}
    stop_words = set(['、', '：', '。', '，', '的', '等', '一', '二',
                     '三', '（', '）', '《', '》', '“', '”', ' ', '\n'])

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            text = f.read()
            for word in jieba.cut(text.strip()):
                if word not in stop_words and word.strip():
                    if word in frequency:
                        frequency[word] += 1
                    else:
                        frequency[word] = 1
                    tokens.append(word.strip())
    return tokens


batch_size = 128
epoch = 100
window = 4
embedding_dim = 100
tokens = read_dir('./data')
dataset = TextReader(tokens, window)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = CBow(dataset.vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epoch):
    total_loss = 0
    model.train()
    for i, (context, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"epoch: {epoch}, loss: {total_loss}")


total = 0
correct = 0
model.eval()
for context_tensor, target_tensor in dataset:
    output = model(context_tensor)
    output_index = output[0].argmax().item()
    correct += 1 if output_index == target_tensor.item() else 0
    total += 1
    print(
        f"input: {[dataset.idx2word[idx] for idx in  context_tensor.tolist()]}, output: {dataset.idx2word[output_index]}, target: {dataset.idx2word[target_tensor.item()]}")

print(f"accuracy: {correct / total:.4f}")
