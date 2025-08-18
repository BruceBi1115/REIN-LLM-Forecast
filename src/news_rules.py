import pandas as pd
import numpy as np
from datetime import timedelta
import pytz
from typing import List

def load_news(path: str, time_col: str, tz: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(pytz.timezone(tz))
    return df

def get_candidates(news_df: pd.DataFrame, time_col: str, target_time, window_days: int, topM: int) -> pd.DataFrame:
    start = target_time - pd.Timedelta(days=window_days)
    cand = news_df[(news_df[time_col] >= start) & (news_df[time_col] < target_time)]
    cand = cand.sort_values(time_col, ascending=False)
    if len(cand) > topM:
        cand = cand.iloc[:topM]
    return cand

def _load_keywords(path: str) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [w.strip().lower() for w in f if w.strip()]
    except FileNotFoundError:
        return []

def policy_recent_topk(cand: pd.DataFrame, K: int) -> pd.DataFrame:
    return cand.head(K)

def policy_by_keywords(cand: pd.DataFrame, text_col: str, keywords: List[str], K: int) -> pd.DataFrame:
    if not keywords:
        return cand.head(K)
    m = cand[text_col].fillna('').astype(str).str.lower()
    mask = m.apply(lambda x: any(kw in x for kw in keywords))
    filtered = cand[mask]
    if len(filtered) == 0:
        return cand.head(K)
    return filtered.head(K)

def policy_mixed_alpha(cand: pd.DataFrame, text_col: str, policy_keys: List[str], sd_keys: List[str], K: int) -> pd.DataFrame:
    scores = np.zeros(len(cand), dtype=float)
    m = cand[text_col].fillna('').astype(str).str.lower().values
    for i, txt in enumerate(m):
        s = 0
        if any(kw in txt for kw in policy_keys): s += 1.0
        if any(kw in txt for kw in sd_keys): s += 1.0
        scores[i] = s
    order = np.argsort(-scores)
    cand2 = cand.iloc[order]
    return cand2.head(K)

def select_news(cand: pd.DataFrame, policy: str, text_col: str,
                policy_kw: List[str], sd_kw: List[str], K: int) -> pd.DataFrame:
    if policy == 'recent_topk':
        return policy_recent_topk(cand, K)
    if policy == 'policy_only':
        return policy_by_keywords(cand, text_col, policy_kw, K)
    if policy == 'supply_demand':
        return policy_by_keywords(cand, text_col, sd_kw, K)
    if policy == 'mixed_alpha':
        return policy_mixed_alpha(cand, text_col, policy_kw, sd_kw, K)
    return policy_recent_topk(cand, K)

def lead3(text: str, max_sentences: int = 3) -> str:
    seps = ['。', '.', '；', ';', '！', '!', '？', '?', '\n']
    tmp = text
    for s in seps:
        tmp = tmp.replace(s, '.')
    parts = [p.strip() for p in tmp.split('.') if p.strip()]
    return ' '.join(parts[:max_sentences])
