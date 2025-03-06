import torch
from typing import Literal


def get_device() -> Literal["mps", "cuda", "cpu"]:
    # if torch.backends.mps.is_available():  # type: ignore
    #     return 'mps'
    # elif torch.cuda.is_available():
    #     return 'cuda'
    # else:
    #     return 'cpu'

    return 'cpu'
