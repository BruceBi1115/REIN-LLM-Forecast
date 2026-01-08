# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

END_TOKEN = "<END>"
PRED_TOKEN_PREFIX = "<PRED_"


def build_pred_slots(horizon: int) -> str:
    return " ".join([f"<PRED_{i}>" for i in range(int(horizon))])


class TSForecastRegressor(nn.Module):
    """
    Text + (time-series patches as soft tokens) -> LLM -> pool patch hidden states -> regression head -> H values

    - input_ids / attention_mask: tokenized prompt (instruction + news + any brief history text)
    - ts_patches: float tensor (B, P, patch_dim)  (z-scored patches)
    - ts_patch_mask: 0/1 mask (B, P) for padded patches
    - targets: float tensor (B, H) (z-scored targets). If provided, returns MSE loss in z-space.
    """

    def __init__(
        self,
        lm,
        horizon: int,
        patch_dim: int,
        hidden_size: int,
        patch_dropout: float = 0.0,
        head_dropout: float = 0.0,
        head_mlp: bool = False,
    ):
        super().__init__()
        self.lm = lm
        self.horizon = int(horizon)
        self.patch_dim = int(patch_dim)
        self.hidden_size = int(hidden_size)

        self.patch_proj = nn.Linear(self.patch_dim, self.hidden_size)
        self.patch_drop = nn.Dropout(float(patch_dropout))

        if head_mlp:
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(float(head_dropout)),
                nn.Linear(self.hidden_size, self.horizon),
            )
        else:
            self.head_drop = nn.Dropout(float(head_dropout))
            self.head = nn.Linear(self.hidden_size, self.horizon)

        lm_dtype = next(self.lm.parameters()).dtype
        self.patch_proj = self.patch_proj.to(dtype=lm_dtype)
        self.head = self.head.to(dtype=lm_dtype)

    def _pool_patch_hidden(self, last_hidden: torch.Tensor, ts_patch_mask: torch.Tensor) -> torch.Tensor:
        """
        last_hidden: (B, T_text + P, H)
        ts_patch_mask: (B, P) 0/1
        returns pooled: (B, H)
        """
        B, seq_len, H = last_hidden.shape
        P = ts_patch_mask.size(1)
        patch_hid = last_hidden[:, -P:, :]  # (B, P, H)

        m = ts_patch_mask.to(dtype=patch_hid.dtype).unsqueeze(-1)  # (B, P, 1)
        denom = m.sum(dim=1).clamp_min(1.0)  # (B, 1)

        pooled = (patch_hid * m).sum(dim=1) / denom  # (B, H)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ts_patches: torch.Tensor,
        ts_patch_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        # text token embeddings (dtype usually bf16)
        tok_emb = self.lm.get_input_embeddings()(input_ids)  # (B, T, H)
        tok_dtype = tok_emb.dtype

        # ---关键：让 patch 分支 dtype 对齐---
        proj_dtype = self.patch_proj.weight.dtype
        if ts_patches.dtype != proj_dtype:
            ts_patches = ts_patches.to(dtype=proj_dtype)

        patch_emb = self.patch_proj(ts_patches)  # (B, P, H)  dtype = proj_dtype
        patch_emb = self.patch_drop(patch_emb)

        # 再对齐到 tok_emb 的 dtype，保证 cat 后 inputs_embeds 统一 dtype
        if patch_emb.dtype != tok_dtype:
            patch_emb = patch_emb.to(dtype=tok_dtype)

        # concat as "soft tokens": [text_tokens, patch_tokens]
        inputs_embeds = torch.cat([tok_emb, patch_emb], dim=1)  # (B, T+P, H)
        attn = torch.cat([attention_mask, ts_patch_mask], dim=1)  # (B, T+P)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]  # (B, T+P, H)
        pooled = self._pool_patch_hidden(last_hidden, ts_patch_mask)  # (B, H)

        if isinstance(self.head, nn.Linear):
            pred = self.head(self.head_drop(pooled))
        else:
            pred = self.head(pooled)

        out = {"pred": pred}

        if targets is not None:
            # targets 建议也对齐 pred dtype（避免 bf16/float32 混算警告）
            if targets.dtype != pred.dtype:
                targets = targets.to(dtype=pred.dtype)
            loss = F.mse_loss(pred, targets, reduction="mean")
            out["loss"] = loss

        return out


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
    patch_dim: int = 4,
    patch_dropout: float = 0.0,
    head_dropout: float = 0.0,
    head_mlp: bool = False,
):
    tok = AutoTokenizer.from_pretrained(tokenizer_id or base_model, use_fast=True)

    # pad token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # (Optional) keep END token for backward-compat; not used by regressor
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
    lm = get_peft_model(base, peft_cfg)
    lm.print_trainable_parameters()

    hidden_size = lm.config.hidden_size

    model = TSForecastRegressor(
        lm=lm,
        horizon=int(horizon),
        patch_dim=int(patch_dim),
        hidden_size=int(hidden_size),
        patch_dropout=float(patch_dropout),
        head_dropout=float(head_dropout),
        head_mlp=bool(head_mlp),
    )

    return tok, model
