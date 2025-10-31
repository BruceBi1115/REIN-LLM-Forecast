#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_by_ratio.py
按给定比例将 CSV 切分为 train/val/test 三个子集。

用法示例：
1) 按 7:2:1 切分（最常见）：
   python split_by_ratio.py data.csv 7 2 1

2) 也支持 70 20 10 或 0.7 0.2 0.1，都会自动归一化：
   python split_by_ratio.py data.csv 70 20 10

可选参数：
   --seed 42           随机种子（默认 42）
   --no-shuffle        不打乱（默认会打乱）
   --output-dir ./out  输出目录（默认当前目录）
   --encoding utf-8    文件编码（默认让 pandas 自动判断）
"""

import os
import argparse
import math
import numpy as np
import pandas as pd


def normalize_ratios(a, b, c):
    r = np.array([a, b, c], dtype=float)
    if (r < 0).any():
        raise ValueError("比例不能为负数。")
    s = r.sum()
    if s <= 0:
        raise ValueError("比例之和必须大于 0。")
    r = r / s  # 归一化为和为 1
    return float(r[0]), float(r[1]), float(r[2])


def split_indices(n, tr, va, seed=42, shuffle=True):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    if shuffle:
        rng.shuffle(idx)
    n_train = int(math.floor(n * tr))
    n_val = int(math.floor(n * va))
    n_test = n - n_train - n_val  # 剩余全部给 test，确保总数不丢
    i_tr = idx[:n_train]
    i_va = idx[n_train:n_train + n_val]
    i_te = idx[n_train + n_val:]
    return i_tr, i_va, i_te


def main():
    ap = argparse.ArgumentParser(description="按比例将 CSV 切分为 train/val/test")
    ap.add_argument("input", help="输入 CSV 路径")
    ap.add_argument("train", type=float, help="train 比例（如 7 或 0.7）")
    ap.add_argument("val", type=float, help="val 比例（如 2 或 0.2）")
    ap.add_argument("test", type=float, help="test 比例（如 1 或 0.1）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                    help="不随机打乱（默认会打乱）")
    ap.add_argument("--output-dir", default=".", help="输出目录（默认当前目录）")
    ap.add_argument("--encoding", default=None,
                    help="读取文件编码（如 'utf-8', 'gbk'），默认 None 自动判断")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"找不到输入文件：{args.input}")

    tr, va, te = normalize_ratios(args.train, args.val, args.test)
    df = pd.read_csv(args.input, encoding=args.encoding)
    n = len(df)
    if n == 0:
        raise ValueError("输入 CSV 为空。")

    i_tr, i_va, i_te = split_indices(n, tr, va, seed=args.seed, shuffle=args.shuffle)

    base = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    out_train = os.path.join(args.output_dir, f"{base}_trainset.csv")
    out_val = os.path.join(args.output_dir, f"{base}_valset.csv")
    out_test = os.path.join(args.output_dir, f"{base}_testset.csv")

    df.iloc[i_tr].to_csv(out_train, index=False)
    df.iloc[i_va].to_csv(out_val, index=False)
    df.iloc[i_te].to_csv(out_test, index=False)

    print(f"总样本数: {n}")
    print(f"Train({tr:.4f}): {len(i_tr)} → {out_train}")
    print(f"Val  ({va:.4f}): {len(i_va)} → {out_val}")
    print(f"Test ({te:.4f}): {len(i_te)} → {out_test}")
    print("完成。")


if __name__ == "__main__":
    main()
