# gpt_client.py
from __future__ import annotations
import json
import os
import time
from typing import Dict, Any, Optional, List, Set, Iterator
from string import Formatter
from pathlib import Path
from openai import OpenAI, APIConnectionError, RateLimitError, BadRequestError

class ConfigError(Exception):
    pass
def update_total_cost(file_path: str, add_value: float) -> float:
    """
    读取 file_path 中已有的消费总额，加上 add_value，再写回文件。
    返回新的总额。
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 读旧值
    try:
        old_val = float(path.read_text().strip())
    except Exception:
        old_val = 0.0

    new_val = old_val + add_value

    # 保存新值（保留 6 位小数）
    path.write_text(f"{new_val:.12f}", encoding="utf-8")
    return new_val

USD_TO_AUD = 1.5
def calc_cost(usage: dict) -> float:
    d = usage.model_dump()
    """返回本次调用费用（澳币）"""
    in_tok = d.get("input_tokens", 0)
    out_tok = d.get("output_tokens", 0)
    usd = (in_tok / 1_000_000) * 0.15 + (out_tok / 1_000_000) * 0.60
    return usd * USD_TO_AUD

class PromptManager:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "prompts" not in cfg or not isinstance(cfg["prompts"], dict):
            raise ConfigError("config.json 缺少 prompts 字段或格式不正确。")
        self.cfg = cfg
        self.prompts: Dict[str, str] = cfg["prompts"]
        self.model: str = cfg.get("model", "gpt-4o-mini")
        # API key 优先：环境变量 > config.json
        self.api_key: str = os.getenv("OPENAI_API_KEY") or cfg.get("openai_api_key") or ""
        if not self.api_key:
            raise ConfigError("未找到 API Key。请设置环境变量 OPENAI_API_KEY 或在 config.json 的 openai_api_key 中填写。")

    def available_types(self) -> List[str]:
        return sorted(self.prompts.keys())

    @staticmethod
    def _extract_placeholders(tmpl: str) -> Set[str]:
        names = set()
        for _, field_name, _, _ in Formatter().parse(tmpl):
            if field_name:
                simple = field_name.split("[")[0].split(".")[0]
                names.add(simple)
        return names

    def render(self, kind: str, variables: Dict[str, Any]) -> str:
        if kind not in self.prompts:
            raise ConfigError(f"未找到模板类型 '{kind}'。可选：{self.available_types()}")
        tmpl = self.prompts[kind]
        required = self._extract_placeholders(tmpl)
        missing = [k for k in required if k not in variables]
        if missing:
            raise ConfigError(f"模板 '{kind}' 缺少必需变量：{missing}")
        try:
            return tmpl.format(**variables)
        except KeyError as e:
            raise ConfigError(f"变量缺失：{e!s}")
        except Exception as e:
            raise ConfigError(f"渲染模板失败：{e}")

class ChatClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout: float = 60.0):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.timeout = timeout

    def run_prompt(
        self,
        rendered_prompt: str,
        system_text: Optional[str] = None,
        temperature: float = 0.3,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
    ) -> str:
        messages = []
        if system_text:
            messages.append({"role": "system", "content": [{"type": "input_text", "text": system_text}]})
        messages.append({"role": "user", "content": [{"type": "input_text", "text": rendered_prompt}]})

        attempt = 0
        while True:
            try:
                rsp = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    temperature=temperature,
                    timeout=self.timeout,
                )
                if hasattr(rsp, "usage"):
                    cost_aud = calc_cost(rsp.usage)
                    print("[Cost] AUD$%.6f" % cost_aud)
                    total = update_total_cost("cost.txt", cost_aud)
                    print("[Total Cost] AUD$%.6f" % total)
                    

                if hasattr(rsp, "output") and rsp.output:
                    for out in rsp.output:
                        if out.type == "message":
                            parts = out.content or []
                            text_chunks = [p.text for p in parts if getattr(p, "type", "") in ("output_text", "text")]
                            if text_chunks:
                                return "".join(text_chunks).strip()
                return (getattr(rsp, "output_text", None) or "").strip()
            except (APIConnectionError, RateLimitError):
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(retry_backoff ** attempt)
            except BadRequestError:
                raise

    def stream_prompt(
        self,
        rendered_prompt: str,
        system_text: Optional[str] = None,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        messages = []
        if system_text:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": rendered_prompt}]})

        with self.client.responses.stream(
            model=self.model,
            input=messages,
            temperature=temperature,
            timeout=self.timeout,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
            stream.final_message()

def load_client_from_config(config_path: str) -> tuple[PromptManager, ChatClient]:
    pm = PromptManager(config_path)
    cc = ChatClient(api_key=pm.api_key, model=pm.model)
    return pm, cc

# ===== 新增：可在外部 py 直接调用的一行函数 =====

def run_from_config(
    config_path: str,
    kind: str,
    variables: Dict[str, Any],
    system: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    """
    渲染指定模板并调用模型，返回完整字符串结果。
    """
    pm, cc = load_client_from_config(config_path)
    rendered = pm.render(kind, variables)
    return cc.run_prompt(rendered, system_text=system, temperature=temperature)

def stream_from_config(
    config_path: str,
    kind: str,
    variables: Dict[str, Any],
    system: Optional[str] = None,
    temperature: float = 0.3,
) -> Iterator[str]:
    """
    渲染指定模板并以流式方式调用模型，返回一个逐字增量的迭代器。
    """
    pm, cc = load_client_from_config(config_path)
    rendered = pm.render(kind, variables)
    return cc.stream_prompt(rendered, system_text=system, temperature=temperature)
