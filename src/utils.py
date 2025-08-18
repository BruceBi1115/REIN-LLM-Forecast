import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_from_id(gpu_id: int):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')

def count_tokens(tokenizer, text: str):
    return len(tokenizer.encode(text, add_special_tokens=False))
