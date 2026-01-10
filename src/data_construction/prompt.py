import yaml

from src.news_rules import lead3
from ..utils.utils import count_tokens
import numpy as np
import torch
from typing import List


def load_templates(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    tpls = {t['id']: t for t in cfg['templates']}
    return tpls

def format_history_cutoff(hist_values, unit: str, budget_tokens: int, tokenizer):
    text = ', '.join([f'{float(v):.4f}' for v in hist_values])
    while count_tokens(tokenizer, text) > budget_tokens and len(hist_values) > 4:
        hist_values = hist_values[2:]
        text = ', '.join([f'{float(v):.4f}' for v in hist_values])
    return text

# def format_news(news_rows, text_col: str, budget_tokens: int, tokenizer, summary_method='lead3', max_sentences=3):
#     bullets = []
#     for _, r in news_rows.iterrows():
#         txt = str(r.get(text_col, ''))
#         if summary_method == 'lead3':
#             txt = lead3(txt, max_sentences=max_sentences)
#         bullets.append(f"- {txt}")
#     text = '\n'.join(bullets)
#     while count_tokens(tokenizer, text) > budget_tokens and len(bullets) > 1:
#         bullets = bullets[:-1]
#         text = '\n'.join(bullets)
#     return text

def format_news(
    news_rows,
    text_col: str,
    budget_tokens: int,
    tokenizer,
    summary_method: str = "none",
    max_sentences: int = 3,
):
    """
    Version without:
      - lead3 / any summarization
      - token-budget truncation loop

    It simply formats each row as a bullet using the raw text in `text_col`.

    Notes:
      - `budget_tokens`, `summary_method`, `max_sentences` are kept for API compatibility,
        but are not used.
      - `news_rows` is expected to be a pandas DataFrame (used via .iterrows()).
    """
    bullets = []
    for _, r in news_rows.iterrows():
        txt = str(r.get(text_col, "") or "")
        bullets.append(f"- {txt}")
    return "\n".join(bullets)

def build_prompt(template_text: str, L: int, H: int, unit: str, description: str,
                 history_str: str, news_str: str, value_col, freq, start_date, end_date, pred_start, pred_end, region) -> str:
    prompt = template_text.replace('{L}', str(L)).replace('{H}', str(H)).replace('{UNIT}', unit)
    prompt = prompt.replace('{HISTORY}', history_str) \
    .replace('{NEWS}', news_str).replace('{value_col}', value_col) \
    .replace('{freq}', str(freq)).replace('{start_date}', start_date).replace('{end_date}', end_date)\
    .replace('{pred_start}', pred_start).replace('{pred_end}', pred_end).replace("{description}", description).replace("{region}", region)

    # Set breath frequency and n_paths to fixed values for consistency
    prompt = prompt.replace("{breath_freq}", str(10)).replace("{n_paths}", str(5))
    return prompt
