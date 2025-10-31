import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

SPECIAL_PRED_TOKEN = "<PRED>"

class RegressionHeadModel(nn.Module):
    def __init__(self, base_lm, hidden_size: int, horizon: int):
        super().__init__()
        self.lm = base_lm
        self.horizon = horizon

        base_dtype = next(base_lm.parameters()).dtype   # 通常是 torch.bfloat16（或 float16/float32）
        self.reg_head = torch.nn.Linear(hidden_size, horizon, bias=True)
        self.reg_head = self.reg_head.to(base_dtype)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        pred_token_id = self.lm.config.pred_token_id
        pred_mask = (input_ids == pred_token_id)
        if not pred_mask.any(dim=1).all():
            # 给出更友好的报错
            bad_rows = (~pred_mask.any(dim=1)).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"No <PRED> token found in rows: {bad_rows}")
        idx = pred_mask.float().argmax(dim=1)
        B = input_ids.size(0)
        h_vec = hidden[torch.arange(B), idx, :]
        h_vec = h_vec.to(self.reg_head.weight.dtype)
        y_hat = self.reg_head(h_vec)
        loss = None
        if labels is not None:
            # 计算MSE损失
            loss = nn.functional.mse_loss(y_hat, labels)
        return {'loss': loss, 'pred': y_hat}

def load_llama_lora(base_model: str, tokenizer_id: str, lora_r: int, lora_alpha: int,
                    lora_dropout: float, target_modules, load_in_4bit=False,
                    gradient_checkpointing=False, max_seq_len=1536, device=None, horizon: int = 48):
    tok = AutoTokenizer.from_pretrained(tokenizer_id or base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    if SPECIAL_PRED_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({'additional_special_tokens':[SPECIAL_PRED_TOKEN]})
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if device is None else None,
        load_in_4bit=load_in_4bit
    )
    base.resize_token_embeddings(len(tok))
    base.config.pred_token_id = tok.convert_tokens_to_ids(SPECIAL_PRED_TOKEN)
    if gradient_checkpointing:
        base.gradient_checkpointing_enable()

    peft_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, peft_cfg)
    hidden_size = base.config.hidden_size
    wrapped = RegressionHeadModel(model, hidden_size, horizon=horizon)
    return tok, wrapped
