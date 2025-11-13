#从验证集固定抽一小片（probe）形成稳定的评估子集，减少奖励噪声与评估成本。
import numpy as np
import torch
from torch.utils.data import Subset

def prepare_val_probe_by_number(val_loader, size):
    idxs = list(range(len(val_loader.dataset)))
    np.random.shuffle(idxs)
    idxs = idxs[:min(size, len(idxs))]
    probe_ds = Subset(val_loader.dataset, idxs)
    probe_loader = torch.utils.data.DataLoader(probe_ds, batch_size=val_loader.batch_size, shuffle=True)
    return probe_loader

def prepare_val_probe_by_frac(val_loader, frac: float, *, min_samples: int = 1,
                              seed: int, keep_order: bool = True):
    """
    从验证集按比例抽取一个固定 probe 子集（不放回抽样）。
    - frac ∈ (0, 1]：抽取比例
    - min_samples：至少抽多少个
    - seed：为复现实验可指定随机种子
    - keep_order：是否按原数据集顺序排序抽取到的索引（便于日志/复现）
    """
    assert 0 < frac <= 1.0, "frac 必须在 (0, 1] 之间"
    dataset = val_loader.dataset
    N = len(dataset)

    # 计算抽样数目
    k = max(min_samples, int(np.floor(N * frac)))
    k = min(k, N)

    rng = np.random.default_rng(seed)
    idxs = rng.choice(N, size=k, replace=False)
    if keep_order:
        idxs = np.sort(idxs)

    probe_ds = Subset(dataset, idxs.tolist())
    probe_loader = torch.utils.data.DataLoader(
        probe_ds,
        batch_size=val_loader.batch_size,
        shuffle=True
    )
    return probe_loader
