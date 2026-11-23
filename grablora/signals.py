from __future__ import annotations

import torch


def weight_grads(model, target_mods: dict, batch: dict) -> dict:
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    grads = {}
    for k, m in target_mods.items():
        grads[k] = m.weight.grad.detach().float().clone()
        m.weight.grad = None                       # free the .grad slot, don't accumulate
    return grads, float(out.loss.detach())


def per_sample_weight_grads(model, target_mods: dict, micro_batches: list) -> dict:

    out = {k: [] for k in target_mods}
    for b in micro_batches:
        g, _ = weight_grads(model, target_mods, b)
        for k in target_mods:
            out[k].append(g[k].cpu())              # CPU storage; bg moves back per layer
    return out


@torch.no_grad()
def set_weight(module, W_math: torch.Tensor) -> None:
    module.weight.data.copy_(W_math.to(module.weight.data))   # match dtype AND device


@torch.no_grad()
def add_weight(module, dW_math: torch.Tensor) -> None:
    module.weight.data.add_(dW_math.to(module.weight.data))   # dW may be CPU; move to weight