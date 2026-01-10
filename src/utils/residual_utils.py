# src/residual_utils.py
from __future__ import annotations

import math
import torch
import torch.nn as nn


def freeze_module(m: nn.Module) -> None:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False


def set_trainable(m: nn.Module) -> None:
    m.train()
    for p in m.parameters():
        p.requires_grad = True


def zero_regressor_head(model: nn.Module) -> None:
    """
    Make the regressor output start near 0, so the delta-model naturally predicts small residuals initially.
    Works for:
      - model.head is nn.Linear
      - model.head is nn.Sequential(..., nn.Linear)
    """
    head = getattr(model, "head", None)
    if head is None:
        return

    def _zero_linear(l: nn.Linear):
        nn.init.zeros_(l.weight)
        if l.bias is not None:
            nn.init.zeros_(l.bias)

    if isinstance(head, nn.Linear):
        _zero_linear(head)
        return

    # if head is a sequential, try to zero the last Linear
    if isinstance(head, nn.Sequential):
        for layer in reversed(head):
            if isinstance(layer, nn.Linear):
                _zero_linear(layer)
                return


def split_two_stage_epochs(total_epochs: int, base_frac: float = 0.3, min_base: int = 1, min_delta: int = 1):
    total_epochs = int(total_epochs)
    if total_epochs <= 0:
        return 0, 0
    base_epochs = max(min_base, int(math.floor(total_epochs * float(base_frac))))
    base_epochs = min(base_epochs, total_epochs - min_delta)
    delta_epochs = total_epochs - base_epochs
    if delta_epochs < min_delta:
        delta_epochs = min_delta
        base_epochs = max(0, total_epochs - delta_epochs)
    return base_epochs, delta_epochs
