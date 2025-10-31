import pandas as pd
import numpy as np
from datetime import timedelta
import pytz
from typing import List
from numpy.linalg import norm
from dataclasses import dataclass
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
# ---------- 1) 文本编码器：优先 SBERT，备选 TF-IDF ----------
class _SBERTEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        
        self.m = SentenceTransformer(model_name)
    def fit(self, texts): return self
    def encode(self, texts):
        return self.m.encode(list(texts), normalize_embeddings=True)

class _TFIDFEncoder:
    def __init__(self, max_features=20000):
        
        self.vec = TfidfVectorizer(max_features=max_features)
        self.fitted = False
    def fit(self, texts):
        self.vec.fit(list(texts)); self.fitted = True; return self
    def encode(self, texts):
        X = self.vec.transform(list(texts))
        X = X.astype(np.float32).toarray()
        # L2 normalize
        n = np.maximum(1e-8, np.linalg.norm(X, axis=1, keepdims=True))
        return X / n

@dataclass
class NewsEncoder:
    backend: str = "auto"   # "auto"|"sbert"|"tfidf"
    _enc: object = None
    _cache_key: str = ""

    def build(self, corpus_texts, force_rebuild=False):
        key = f"{self.backend}|{len(corpus_texts)}"
        if (not force_rebuild) and self._enc is not None and key == self._cache_key:
            return self
        try:
            if self.backend in ("auto", "sbert"):
                self._enc = _SBERTEncoder()
                self._enc.fit(corpus_texts)
                self.backend = "sbert"
            else:
                raise RuntimeError()
        except Exception:
            self._enc = _TFIDFEncoder(max_features=20000).fit(corpus_texts)
            self.backend = "tfidf"
        self._cache_key = key
        return self

    def encode(self, texts):
        return self._enc.encode(texts)

# ---------- 2) 相似度 & MMR ----------
def _cos_sim_vec_vs_mat(v, M):  # v: (d,), M: (n,d)
    return M @ v

def _mmr(query_vec, cand_mat, topK=5, lam=0.7):
    n = cand_mat.shape[0]
    sims = cand_mat @ query_vec
    chosen, remain = [], list(range(n))
    for _ in range(min(topK, n)):
        best_i, best_score = -1, -1e9
        for i in remain:
            div = 0.0
            if chosen:
                # 与已选集合的最大相似度
                div = float(np.max(cand_mat[chosen] @ cand_mat[i]))
            score = lam * sims[i] - (1.0 - lam) * div
            if score > best_score:
                best_score, best_i = score, i
        chosen.append(best_i)
        remain.remove(best_i)
    return chosen

# ---------- 3) 无关键词策略实现 ----------
def _select_news_semantic(policy_name, cand_df, text_col, topK, encoder: NewsEncoder,
                          query_text: str, now_ts=None, region=None,
                          alpha=0.7, beta=0.2, gamma=0.1, lam=0.7, tz=None, time_col=None):
    texts = cand_df[text_col].fillna("").astype(str).tolist()
    # 编码
    news_mat = np.array(encoder.encode(texts))          # (n,d) 已归一化
    q_vec    = np.array(encoder.encode([query_text]))[0]# (d,)   已归一化

    # 语义分
    s_sem = news_mat @ q_vec                            # (n,)

    # 时间衰减：需要 cand_df[time_col] 是 datetime64
    if time_col is not None and time_col in cand_df.columns and now_ts is not None:
        dt_hours = (now_ts - cand_df[time_col]).dt.total_seconds().to_numpy() / 3600.0
        dt_hours = np.clip(dt_hours, 0, 1e6)
        s_time = np.exp(-dt_hours / 24.0)               # 24 小时半衰示例
    else:
        s_time = np.ones(len(cand_df), dtype=np.float32)

    # 区域加成（示例：title/text 含 region 名称时+boost，实际可改为更严谨的映射）
    if region is not None:
        region = str(region).lower()
        s_reg = cand_df[text_col].str.lower().str.contains(region).astype(np.float32).to_numpy()
    else:
        s_reg = np.zeros(len(cand_df), dtype=np.float32)

    if policy_name == "dense_retrieval":
        score = s_sem
        idx = np.argsort(-score)[:topK]
        return cand_df.iloc[idx]

    if policy_name == "mmr":
        # 先挑一个较大的候选，再 MMR 去重
        topN = min(max(topK*5, topK), len(cand_df))
        idx0 = np.argsort(-s_sem)[:topN]
        chosen_local = _mmr(q_vec, news_mat[idx0], topK=topK, lam=lam)
        return cand_df.iloc[idx0[chosen_local]]

    if policy_name == "hybrid_alpha":
        score = alpha*s_sem + beta*s_time + gamma*s_reg
        # 先综合打分，再做一次 MMR 去重（可选）
        topN = min(max(topK*5, topK), len(cand_df))
        idx0 = np.argsort(-score)[:topN]
        chosen_local = _mmr(q_vec, news_mat[idx0], topK=topK, lam=0.75)
        return cand_df.iloc[idx0[chosen_local]]

    # 默认兜底：按时间最近
    if time_col is not None and time_col in cand_df.columns:
        return cand_df.sort_values(time_col, ascending=False).head(topK)
    return cand_df.head(topK)



def load_news(path: str, time_col: str, tz: str) -> pd.DataFrame:
    """
    Load news ONLY from a JSON file that contains an array of objects.
    - path: path to a .json file
    - time_col: name of the datetime column in the JSON objects
    - tz: target timezone string (e.g., "Australia/Sydney")
    """
    if not path.endswith(".json"):
        raise ValueError(f"Only .json files are supported, got: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Expect the JSON to be a list of records (array of objects)
    df = pd.read_json(path)  # if your file is JSON Lines, use: pd.read_json(path, lines=True)

    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in JSON file.")

    # Parse as UTC-aware then convert to target timezone
    df[time_col] = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce", utc=True)
    # print(df)
    df = df.dropna(subset=[time_col])

    if tz:
        df[time_col] = df[time_col].dt.tz_convert(tz)
    

    # print(df.sort_values(time_col).reset_index(drop=True))

    return df.sort_values(time_col).reset_index(drop=True)


def _align_ts_to_series_tz(ts, ref_series: pd.Series) -> pd.Timestamp:
    """将 ts 解析为 pandas.Timestamp，并对齐到 ref_series 的时区（或去掉时区）。"""
    
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return ts
    # 取新闻列的时区（可能为 None）
    tz = getattr(ref_series.dt, "tz", None)
    # print("ref_series tz:", tz)
    # print("ts.tzinfo = ",ts.tzinfo)
    if tz is not None:
        # ref 是 tz-aware
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
    else:
        # ref 是 tz-naive
        if ts.tzinfo:
            ts = ts.tz_convert("UTC").tz_localize(None)  # 统一转成 naive
    return ts

def get_num_news_between(news_df, time_col, target_time, window_days):
    # ✅ 先把传入的 target_time 对齐到新闻列的时区
    target_time = _align_ts_to_series_tz(target_time, news_df[time_col])

    if pd.isna(target_time):
        return 0  # 无效时间

    start = target_time - pd.Timedelta(days=window_days)
    count = news_df[(news_df[time_col] >= start) & (news_df[time_col] < target_time)].shape[0]
    return count

def get_candidates(news_df, time_col, target_time, window_days, topM):
    # ✅ 先把传入的 target_time 对齐到新闻列的时区
    # print("original target_time:", target_time)
    target_time = _align_ts_to_series_tz(target_time, news_df[time_col])

    # print("Target time aligned to news timezone:", target_time)
    if pd.isna(target_time):
        return news_df.iloc[0:0]  # 空 DataFrame：没有候选

    start = target_time - pd.Timedelta(days=window_days)
    # print("Start time for candidates:", start)
    
    cand = news_df[(news_df[time_col] >= start) & (news_df[time_col] < target_time)]
    # print(f"Found {len(cand)} candidates in the window from {start} to {target_time}")
    # 取最近 topM 条
    return cand.sort_values(time_col, ascending=False).head(topM)

def _load_keywords(path: str) -> List[str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [w.strip().lower() for w in f if w.strip()]
    except FileNotFoundError:
        return []
    return cand.head(K)

def policy_by_keywords(cand: pd.DataFrame, text_col: str, keywords: List[str], K: int) -> pd.DataFrame:
    if not keywords:
        return cand.head(K)
    m = cand[text_col].fillna('').astype(str).str.lower()
    mask = m.apply(lambda x: any(kw.strip() in x for kw in keywords))
    filtered = cand[mask]
    if len(filtered) == 0:
        return cand.head(K)
    return filtered.head(K)

def select_by_sentiment(cand: pd.DataFrame, text_col: str, K: int) -> pd.DataFrame:
    if len(cand) == 0:
        return cand
    sentiments = cand[text_col].fillna('').astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    order = np.argsort(-sentiments.values)
    return cand.iloc[order].head(K)

def keyword_sentiment_hybrid(cand: pd.DataFrame, text_col: str, keywords, K: int) -> pd.DataFrame:
    if len(cand) == 0 or K <= 0:
        return cand.head(0)

    s = cand[text_col].fillna('').astype(str)
    sentiments = s.apply(lambda x: TextBlob(x).sentiment.polarity)

    # 关键词命中
    if keywords:
        kws = [kw.strip().lower() for kw in keywords if kw and kw.strip()]
        mask = s.str.lower().apply(lambda x: any(kw in x for kw in kws))
    else:
        mask = pd.Series(False, index=cand.index)

    # 先在命中集内按情感排序
    idx_pref = sentiments[mask].sort_values(ascending=False).index.tolist()
    chosen = idx_pref[:K]

    # 不足则从非命中集里按情感排序补齐
    if len(chosen) < K:
        need = K - len(chosen)
        idx_rest = sentiments[~mask].sort_values(ascending=False).index.tolist()
        chosen += idx_rest[:need]

    return cand.loc[chosen]

def select_news(cand: pd.DataFrame, policy: str, text_col: str,
                policy_kw: List[str], K: int, time_col, query_text=None,now_ts=None, region=None,**kwargs) -> pd.DataFrame:
    if policy == 'keywords':
        return policy_by_keywords(cand, text_col, policy_kw, K)
    if policy == 'sentiment':
        return select_by_sentiment(cand, text_col, K)
    if policy == 'keyword_sentiment_hybrid':
        return keyword_sentiment_hybrid(cand, text_col, policy_kw, K)
    
    # 新增：无关键词策略
    if policy in ("dense_retrieval", "mmr", "hybrid_alpha"):
        if encoder is None:
            encoder = NewsEncoder(backend="auto").build(cand[text_col].fillna("").astype(str).tolist())
        return _select_news_semantic(policy, cand, text_col, K, encoder,
                                     query_text=query_text or "",
                                     now_ts=now_ts, region=region,
                                     alpha=kwargs.get("alpha", 0.7),
                                     beta=kwargs.get("beta", 0.2),
                                     gamma=kwargs.get("gamma", 0.1),
                                     lam=kwargs.get("lam", 0.7),
                                     tz=kwargs.get("tz", None),
                                     time_col=time_col)
    
    return cand

def lead3(text: str, max_sentences: int = 3) -> str:
    seps = ['。', '.', '；', ';', '！', '!', '？', '?', '\n']
    tmp = text
    for s in seps:
        tmp = tmp.replace(s, '.')
    parts = [p.strip() for p in tmp.split('.') if p.strip()]
    return ' '.join(parts[:max_sentences])
