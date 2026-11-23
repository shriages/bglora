from __future__ import annotations

import math


def make_schedule(kind: str = "cosine", total: int = 1, warmup: int = 0,
                  final_frac: float = 0.0):
    warmup = max(int(warmup), 0)
    total = max(int(total), 1)

    def f(t: int) -> float:
        if warmup and t < warmup:
            return (t + 1) / warmup
        p = (t - warmup) / max(total - warmup, 1)
        p = min(max(p, 0.0), 1.0)
        if kind == "cosine":
            return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * p))
        if kind == "linear":
            return final_frac + (1 - final_frac) * (1 - p)
        return 1.0

    return f