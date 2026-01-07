import random
import numpy as np
import torch
import pandas as pd

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def device_from_id(gpu_id: int):
    print("torch version:", torch.__version__)
    print("compiled with CUDA:", torch.version.cuda)
    print("cuda.is_available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    print(torch.version.cuda)
    if torch.cuda.is_available():
        print(f'Using GPU id: {gpu_id}')
        return torch.device(f'cuda:{gpu_id}')
    else:
        print('Using CPU')
        return torch.device('cpu')

def count_tokens(tokenizer, text: str):
    return len(tokenizer.encode(text, add_special_tokens=False))

def compute_volatility_bin(df, time_col="", value_col="", window=48, bins=10, dayfirst=True):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=dayfirst)
    df = df.sort_values(time_col)

    recent = df[value_col].iloc[-window:]
    if len(recent) < 2:
        return 0

    vol = recent.std()
    all_std = df[value_col].rolling(window).std().dropna()
    if len(all_std) == 0:
        return 0

    thresholds = np.quantile(all_std, np.linspace(0, 1, bins + 1)[1:-1])
    bin_id = np.digitize(vol, thresholds, right=True)
    return int(min(bin_id, bins - 1))