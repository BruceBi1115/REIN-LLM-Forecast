import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SlidingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, time_col: str, value_col: str,
                 L: int, H: int, stride: int = 1, id_col: str = ''):
        self.time_col = time_col
        self.value_col = value_col
        self.id_col = id_col if id_col else None
        self.L, self.H, self.stride = L, H, stride

        if self.id_col:
            self.groups = []
            for gid, g in df.groupby(self.id_col):
                g = g.sort_values(time_col).reset_index(drop=True)
                self.groups.append((gid, g))
        else:
            g = df.sort_values(time_col).reset_index(drop=True)
            self.groups = [(None, g)]

        self.index = []  # (group_idx, start_idx)
        for gi, (_, g) in enumerate(self.groups):
            n = len(g)
            for s in range(0, n - (L + H) + 1, stride):
                self.index.append((gi, s))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        gi, s = self.index[i]
        gid, g = self.groups[gi]
        window = g.iloc[s:s+self.L]
        target = g.iloc[s+self.L:s+self.L+self.H]
        hist = window[self.value_col].values.astype(np.float32)
        y = target[self.value_col].values.astype(np.float32)
        t_target = target[self.time_col].iloc[-1]
        series_id = gid if gid is not None else 'single'
        return {
            'history': hist,             # (L,)
            'target': y,                 # (H,)
            'target_time': t_target,     # pandas.Timestamp
            'series_id': series_id,
        }

def make_loader(df, time_col, value_col, L, H, stride, batch_size, shuffle=False, id_col=''):
    ds = SlidingDataset(df, time_col, value_col, L, H, stride, id_col)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
