from encoder import get_encoder
from config import GPT2Config
from model import GPT2LMHeadModel
from sample import sample_sequence
from utils import load_weight, compare_models
import argparse
import random
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=False,
                        default="Today is a good day")
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--do_sample", type=bool, default=False)

    args = parser.parse_args()

    if args.quiet is False:
        print(args)

    if args.batch_size == -1:
        args.batch_size = 1

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(
        "/Users/apple/Desktop/python/dh_ai/第二期/week3-gpt2/gpt-2-Pytorch/GPT2/gpt2-pytorch_model.bin", map_location=device)

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = config.n_ctx // 2

    # Generate Text
    context = enc.encode(args.text)
    generated = 0
    while generated < args.nsamples:
        out = sample_sequence(model, length=args.length, context=context,
                              temperature=1, top_k=0, device=device, sample=args.do_sample)  # type: ignore
        out = out[:, len(context):].tolist()
        for i in range(args.batch_size):
            generated += 1
            text = enc.decode(out[i])
            if args.quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)


if __name__ == "__main__":
    main()
