import torch
from model import GPT2LMHeadModel
import torch.nn.functional as F


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def sample_sequence(model, length, batch_size=1, context=None, temperature=1, top_k=0, device='cuda', sample=False):
    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(
        0).repeat(batch_size, 1)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in range(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output


if __name__ == "__main__":
    from mode import GPT2LMHeadModel
    from config import GPT2Config
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    sample_sequence(model, 10, context=[1, 2, 3])
