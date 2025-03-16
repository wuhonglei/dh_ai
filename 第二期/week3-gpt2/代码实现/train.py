import torch
from model import GPT2LMHeadModel
from config import GPT2Config
from encoder import get_encoder
from utils import get_device, load_weight


def train():
    device = get_device()
    text_list = ['how are you']
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    state_dict = torch.load(
        "/Users/apple/Desktop/python/dh_ai/第二期/week3-gpt2/gpt-2-Pytorch/GPT2/gpt2-pytorch_model.bin", map_location=device)
    model = load_weight(model, state_dict)
    model.to(device)

    model.train()
    context = torch.tensor([enc.encode(text)
                           for text in text_list], device=device)
    # 生成 labels: context 左移一位, 最后一个元素替换为 -1
    labels = context.clone()
    labels[:, :-1] = context[:, 1:]
    labels[:, -1] = -1

    output = model(context, lm_labels=labels)
    print('output', output)


if __name__ == "__main__":
    train()
