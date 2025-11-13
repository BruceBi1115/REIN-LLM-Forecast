import numpy as np
from collections import defaultdict

class LinTS:
    def __init__(self, d: int, v: float = 1.0, l2: float = 1.0):
        self.A = l2 * np.eye(d)
        self.b = np.zeros(d)
        self.v = v

    def sample_score(self, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A)
        mu = A_inv @ self.b
        theta = np.random.multivariate_normal(mu, (self.v ** 2) * A_inv)
        return float(theta @ x)

    def update(self, x: np.ndarray, r: float):
        self.A += np.outer(x, x)
        self.b += r * x

class LinUCB:
    def __init__(self, d: int, alpha: float = 1.0, l2: float = 1.0):
        self.A = l2 * np.eye(d)
        self.b = np.zeros(d)
        self.alpha = alpha

    def ucb_score(self, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A)
        mu = A_inv @ self.b
        s = np.sqrt(x @ A_inv @ x)
        return float(mu @ x + self.alpha * s)

    def update(self, x: np.ndarray, r: float):
        self.A += np.outer(x, x)
        self.b += r * x

class RewardNormalizer:
    def __init__(self, ema=0.3, use_group_norm=False):
        self.ema = ema
        self.use_group_norm = use_group_norm
        self.state = defaultdict(lambda: {'mean':0.0,'var':1.0,'count':0})
        self.last_smoothed = 0.0

    def update_and_normalize(self, r: float, group_key=None):
        g = group_key if self.use_group_norm else 'global'
        st = self.state[g]
        st['count'] += 1
        delta = r - st['mean']
        st['mean'] += delta / st['count']
        delta2 = r - st['mean']
        st['var'] = ((st['count']-1)*st['var'] + delta*delta2) / max(1, st['count'])
        std = max(1e-6, np.sqrt(st['var']))
        z = (r - st['mean']) / std
        self.last_smoothed = self.ema * z + (1 - self.ema) * self.last_smoothed
        return self.last_smoothed
