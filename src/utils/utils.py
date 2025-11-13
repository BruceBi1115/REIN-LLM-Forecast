import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_from_id(gpu_id: int):
    print("torch version:", torch.__version__)
    print("compiled with CUDA:", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    print(torch.version.cuda)
    if torch.cuda.is_available():
        print(f'Using GPU: {gpu_id}')
        return torch.device(f'cuda:{gpu_id}')
    else:
        print('Using CPU')
        return torch.device('cpu')

def count_tokens(tokenizer, text: str):
    return len(tokenizer.encode(text, add_special_tokens=False))
