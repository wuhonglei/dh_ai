import os
import shutil


def print_parameters(model):
    param_count = 0
    for name, param in model.named_parameters():
        param_count += param.numel()
        print(name, param.size())
    print(f'param_count: {param_count}')


def print_forward(x, model):
    x = x.view(-1, x.size(1), 1, 1)
    for name, module in model.named_children():
        if name == 'ct':
            continue

        print(f'name: {name}, input: {x.size()}')
        x = module(x)
        print(f'name: {name}, output: {x.size()}')
        print()


def make_dirs(path: str, remove: bool = False):
    if remove and os.path.exists(path):
        shutil.rmtree(path)  # 空目录

    os.makedirs(path, exist_ok=True)
