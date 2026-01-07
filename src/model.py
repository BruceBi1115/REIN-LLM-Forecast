# model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

END_TOKEN = "<END>"
PRED_TOKEN_PREFIX = "<PRED_"

def build_pred_slots(horizon: int) -> str:
    return " ".join([f"<PRED_{i}>" for i in range(int(horizon))])

def load_llama_lora(
    base_model: str,
    tokenizer_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules,
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = False,
    max_seq_len: int = 1536,
    device=None,
    horizon: int = 48,
):
    tok = AutoTokenizer.from_pretrained(tokenizer_id or base_model, use_fast=True)

    # pad token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # add END token as special token
    special = tok.special_tokens_map.get("additional_special_tokens", [])
    if END_TOKEN not in special:
        tok.add_special_tokens({"additional_special_tokens": [END_TOKEN]})

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device is None else None,
        load_in_4bit=load_in_4bit,
    )

    # IMPORTANT: resize embeddings after adding tokens
    base.resize_token_embeddings(len(tok))

    if gradient_checkpointing:
        base.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, peft_cfg)
    model.print_trainable_parameters()

    return tok, model
