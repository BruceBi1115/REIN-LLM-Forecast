import yaml
from .utils import count_tokens

def load_templates(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    tpls = {t['id']: t for t in cfg['templates']}
    return tpls

def format_history(hist_values, unit: str, budget_tokens: int, tokenizer):
    text = ', '.join([f'{float(v):.4f}' for v in hist_values])
    while count_tokens(tokenizer, text) > budget_tokens and len(hist_values) > 4:
        hist_values = hist_values[2:]
        text = ', '.join([f'{float(v):.4f}' for v in hist_values])
    return text

def format_news(news_rows, text_col: str, budget_tokens: int, tokenizer, summary_method='lead3', max_sentences=3):
    from .news_rules import lead3
    bullets = []
    for _, r in news_rows.iterrows():
        txt = str(r.get(text_col, ''))
        if summary_method == 'lead3':
            txt = lead3(txt, max_sentences=max_sentences)
        bullets.append(f"- {txt}")
    text = '\n'.join(bullets)
    while count_tokens(tokenizer, text) > budget_tokens and len(bullets) > 1:
        bullets = bullets[:-1]
        text = '\n'.join(bullets)
    return text

def build_prompt(template_text: str, L: int, H: int, unit: str,
                 history_str: str, news_str: str) -> str:
    prompt = template_text.replace('{L}', str(L)).replace('{H}', str(H)).replace('{UNIT}', unit)
    prompt = prompt.replace('{HISTORY}', history_str).replace('{NEWS}', news_str)
    return prompt
