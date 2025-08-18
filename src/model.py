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
        self.reg_head = nn.Linear(hidden_size, horizon)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        pred_token_id = self.lm.config.pred_token_id
        pred_mask = (input_ids == pred_token_id)
        assert pred_mask.any(), "No <PRED> token found."
        idx = pred_mask.float().argmax(dim=1)
        B = input_ids.size(0)
        h_vec = hidden[torch.arange(B), idx, :]
        y_hat = self.reg_head(h_vec)
        loss = None
        if labels is not None:
            loss = nn.functional.mse_loss(y_hat, labels)
        return {'loss': loss, 'pred': y_hat}

def load_llama_lora(base_model: str, tokenizer_id: str, lora_r: int, lora_alpha: int,
                    lora_dropout: float, target_modules, load_in_4bit=False,
                    gradient_checkpointing=False, max_seq_len=1536, device=None, horizon: int = 48):
    tok = AutoTokenizer.from_pretrained(tokenizer_id or base_model, use_fast=True)
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
