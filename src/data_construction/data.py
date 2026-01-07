import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def fix_date_column(df: pd.DataFrame, time_col: str, keep_time: bool = True, tz: str = None):
    s = df[time_col].astype(str).str.strip()

    # 1) 先按“日/月/年（可带时分秒）”解析（常见：12/01/2024 19:00）
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")

    # 2) 兜底：尝试常见格式（包含年-日-月、年-月-日，带/不带时间）
    for fmt in ("%Y-%d-%m %H:%M:%S", "%Y-%d-%m %H:%M",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                "%d/%m/%Y", "%Y-%d-%m", "%Y-%m-%d"):
        dt = dt.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))

    # 3) 可选：统一到某时区
    if tz:
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(tz)
        else:
            dt = dt.dt.tz_convert(tz)

    # 4) 固定输出格式（字符串）
    mask = dt.isna()
    if keep_time:
        out = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        out = dt.dt.strftime("%Y-%m-%d")
    out = out.mask(mask, "")  # 解析失败的留空

    df[time_col] = out
    return df

class SlidingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str,
        value_col: str,
        L: int,
        H: int,
        stride: int = 1,
        id_col: str = '',
        *,
        # —— 时间解析相关 —— #
        dayFirst: bool,  # 是否是“日-月-年”格式
        parse_time: bool = True,          # 是否在数据集中解析时间列
        tz: str = False,                   # 例如 "Australia/Sydney"；不传则不处理时区
        drop_na_time: bool = True,        # 解析失败的 NaT 是否丢弃
        return_time_iso: bool = True,     # __getitem__ 中是否将 target_time 转为 ISO 字符串
    ):
        """
        df 至少包含时间列与目标值列；可选序列ID列。
        支持“日-月-年”的时间解析、可选的时分拼接(time_str_col)、以及时区本地化/转换。
        """
        self.dayFirst = dayFirst
        self.time_col = time_col
        self.value_col = value_col
        self.id_col = id_col if id_col else None
        self.L, self.H, self.stride = L, H, stride
        self.return_time_iso = return_time_iso

        # === 1) 统一解析时间列（单列，day-first） ===
        if parse_time:
            s = pd.to_datetime(df[time_col], dayfirst=self.dayFirst, errors="coerce")
            # print("s:",s.dt.month.unique())
            # print("s:", s)
            if tz:  # 可选
                if getattr(s.dt, "tz", None) is None:
                    s = s.dt.tz_localize(tz)
                else:
                    s = s.dt.tz_convert(tz)

            # print("s after tz:", s)
            df = df.copy()
            df[time_col] = s
            # print("s:",s.dt.month.unique())

        # 丢掉无效时间/数值并排序（保留原来的逻辑）
        if drop_na_time:
            df = df.dropna(subset=[time_col, value_col])

        # === 2) 分组（多序列）或单序列 ===
        # print(self.id_col)
        # print("df:", df)
        
        if self.id_col:
            self.groups = []
            for gid, g in df.groupby(self.id_col):
                # g = g.sort_values(time_col).reset_index(drop=True)
                self.groups.append((gid, g))
        else:
            # g = df.sort_values(time_col).reset_index(drop=True)
            g = df
            self.groups = [(None, g)]

        # print("groups:", self.groups)

        # === 3) 预生成样本索引 (group_idx, start_idx) ===
        self.index = []
        for gi, (_, g) in enumerate(self.groups):
            n = len(g)
            #print("n:", n, "L:", L, "H:", H, "stride:", stride)
            # 能切出的起点 s: 0..n-(L+H)
            for s in range(0, max(0, n - (L + H)) + 1, stride):
                self.index.append((gi, s))
        #print("index:", self.index)
        # print("g:",g)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        gi, s = self.index[i]
        gid, g = self.groups[gi]

        window = g.iloc[s : s + self.L]
        target = g.iloc[s + self.L : s + self.L + self.H]
        

        hist = window[self.value_col].values.astype(np.float32)   # (L,)
        # print("hist:", hist)
        y    = target[self.value_col].values.astype(np.float32)   # (H,)
        # print("target:", y)

        # 这里的列已经在 __init__ 里解析为 datetime 了，直接格式化
        history_times = window[self.time_col].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("").tolist()
        target_times  = target[self.time_col].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("").tolist()
        t_target_str  = target_times[0] if target_times else ""

        # print("t_target_str:", t_target_str)
        # 假设说有个id column叫region，那series id就可能是nsw，vic等等
        series_id = gid if gid is not None else "Not Specified"
        return {
            "history_value": hist,              # shape (L,) [29.335 26.028 24 ...]
            "target_value": y,                  # shape (H,)
            "history_times": history_times,  # ['2024-01-01 00:30:00', ...]
            "target_times":  target_times,   # ['2024-01-02 00:00:00', ...]
            "target_time":   t_target_str,   # 'YYYY-MM-DD HH:MM:SS'
            "series_id": series_id,
        }

def make_loader(
    df, time_col, value_col, L, H, stride, batch_size, shuffle=False, id_col='',
    **dataset_time_kwargs,
):
    # print(df)
    ds = SlidingDataset(
        df, time_col, value_col, L, H, stride, id_col,
        **dataset_time_kwargs  # 这里把 dayfirst/time_fmt/tz 等透传
    )
    # print("dataset:", ds)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
