import numpy as np

class ValidationState:
    def __init__(self, ema_alpha: float = 0.9):
        self.prev_val_loss: float | None = None
        self.prev_val_loss_ema: float | None = None
        self.ema_alpha = float(ema_alpha)

    def update(self, val_loss: float) -> float:
        """
        更新内部状态；返回本次相对上次的改进（delta>0 表示变好）
        """
        val_loss = float(val_loss)
        delta = 0.0 if self.prev_val_loss is None else float(self.prev_val_loss - val_loss)

        if self.prev_val_loss_ema is None:
            self.prev_val_loss_ema = val_loss
        else:
            a = self.ema_alpha
            self.prev_val_loss_ema = a * self.prev_val_loss_ema + (1.0 - a) * val_loss

        self.prev_val_loss = val_loss
        return delta

    def as_context(self, loss_cap: float = 5.0) -> dict:
        """
        返回用于 encode_instruction 的 E 类上下文，已做截断归一化。
        """
        prev_n = 0.0 if self.prev_val_loss is None else float(np.clip(self.prev_val_loss / loss_cap, 0.0, 1.0))
        ema_n  = 0.0 if self.prev_val_loss_ema is None else float(np.clip(self.prev_val_loss_ema / loss_cap, 0.0, 1.0))
        return {"prev_val_loss_n": prev_n, "prev_val_loss_ema_n": ema_n}
