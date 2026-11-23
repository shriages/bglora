from __future__ import annotations

import torch

from ..methods.manifold import projector_distance


def _subspace_coverage(U, V, G):
    UtG = U.T @ G
    GV = G @ V
    P = U @ UtG + GV @ V.T - U @ (UtG @ V) @ V.T
    return float((P ** 2).sum()) / float((G ** 2).sum() + 1e-30)


def _grad_topr_frac(G, r):
    s = torch.linalg.svdvals(G)
    return float((s[:r] ** 2).sum() / (s ** 2).sum().clamp_min(1e-30))


def layer_records(method, refs, keys, Gfull, start_sub, r) -> list:
    """One row per target layer at the current checkpoint."""
    rows = []
    for i, k in enumerate(keys):
        G = Gfull[k]
        rec = {"key": k, "grad_topr_frac": _grad_topr_frac(G, r)}
        sub = method.subspace(k)
        if sub is not None:
            U, V = sub
            rec["coverage"] = _subspace_coverage(U, V, G)
            if start_sub is not None:
                U0, V0 = start_sub[0][i], start_sub[1][i]
                rec["rotation_t0"] = float(
                    (projector_distance(U, U0) + projector_distance(V, V0))
                    / (2 * (2 * r) ** 0.5))
            else:
                rec["rotation_t0"] = float("nan")
        else:
            rec["coverage"] = float("nan")
            rec["rotation_t0"] = float("nan")
        st = getattr(method, "state", {}).get(k)
        if st is not None and hasattr(st, "m"):          # bg_lora filter state
            rec["rho"] = [round(float(x), 5) for x in st.rho.tolist()]
            rec["m"] = [round(float(x), 5) for x in st.m.tolist()]
            rec["P"] = [round(float(x), 6) for x in st.P.tolist()]
            rec["rho_mean"] = float(st.rho.mean())
            rec["m_abs"] = float(st.m.abs().mean())
        rows.append(rec)
    return rows


def subspace_stack(method, keys):
    Us, Vs = [], []
    for k in keys:
        sub = method.subspace(k)
        if sub is None:
            return None
        Us.append(sub[0]); Vs.append(sub[1])
    return Us, Vs


def compact_state(method, keys) -> dict:
    """Small per-layer adapter state to persist (checkpoints/), never a full model."""
    out = {}
    for k in keys:
        cs = method.__class__.__mro__  # noqa: F841
        st = getattr(method, "state", {}).get(k)
        if st is None:
            continue
        d = {}
        for attr in ("A", "B", "U", "V", "m", "P", "rho"):
            v = getattr(st, attr, None)
            if isinstance(v, torch.Tensor):
                d[attr] = v.detach().float().cpu().clone()
        out[k] = d
    return out