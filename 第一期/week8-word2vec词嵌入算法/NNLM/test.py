import torch

from model import NNLM

checkpoint = torch.load('nnlm.pth')
word2idx = checkpoint['word2idx']
idx2word = checkpoint['idx2word']
context_size = checkpoint['context_size']

model = NNLM(checkpoint['vocab_size'], checkpoint['embedding_dim'],
             checkpoint['hidden_dim'], context_size)
model.load_state_dict(checkpoint['model_state_dict'])

test_words = [
    ['人工', '智能'],
    ['自然语言', '处理'],
    ['人工智能', '具有'],
    ['计算', '领域'],
    ['ai', '计算'],
    ['个性化', '学习'],
    ['计算机', '视觉'],
    ['ai', '计算机']
]

for line in test_words:
    context = [word2idx[word] for word in line if word in word2idx]
    if len(context) < context_size:
        print(f"Context {line} is too short")
        continue
    context = torch.LongTensor(context[-context_size:]).unsqueeze(0)
    output = model.predict(context)
    prob, pred = torch.max(output, 1)
    print(f'{"".join(line)} -> {idx2word[pred.item()]}')
