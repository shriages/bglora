from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..signals import weight_grads, per_sample_weight_grads, set_weight, add_weight


def _topr_full(G, r):
    G = G.detach().float()
    r = min(int(r), min(G.shape))
    q = min(min(G.shape), r + 6)
    U, S, V = torch.svd_lowrank(G, q=q)
    return U[:, :r].contiguous(), S[:r].contiguous(), V[:, :r].contiguous()


def _save_subspaces(save_dir, step, Gfull, keys, r):
    """Write the top-r (U, S, V) of the full gradient for every layer at this step.

    Layout:  <save_dir>/subspaces/<layer>/t<step>.npz   (arrays U, S, V, step)
             <save_dir>/subspaces/meta.json              (rank, per-layer dims, steps)
    One file per (layer, step) -> crash-safe incremental writes, fixed array names."""
    base = os.path.join(str(save_dir), "subspaces")
    os.makedirs(base, exist_ok=True)
    layers_meta = {}
    for k in keys:
        U, S, V = _topr_full(Gfull[k], r)
        ldir = os.path.join(base, str(k).replace("/", "_"))
        os.makedirs(ldir, exist_ok=True)
        np.savez(os.path.join(ldir, f"t{int(step):06d}.npz"),
                 U=U.cpu().numpy(), S=S.cpu().numpy(), V=V.cpu().numpy(),
                 step=np.int64(step))
        layers_meta[str(k)] = [int(U.shape[0]), int(V.shape[0]), int(U.shape[1])]  # d_out, d_in, r
    meta_path = os.path.join(base, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    steps = set(meta.get("steps", [])); steps.add(int(step))
    meta.update({"rank": int(r), "layers": layers_meta,
                 "steps": sorted(steps), "dtype": "float32"})
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _micro_batches(batch: dict, n: int, bs: int, gen) -> list:
    B = batch["input_ids"].shape[0]
    bs = max(1, min(bs, B - 1)) if B > 1 else 1        # bs < B so subsets can differ
    outs = []
    for _ in range(n):
        idx = torch.randperm(B, generator=gen)[:bs]
        outs.append({k: v[idx.to(v.device)] for k, v in batch.items()})
    return outs


def _coverage(method, refs, keys, Gfull) -> float:
    num, den, has_sub = 0.0, 0.0, False
    for k in keys:
        G = Gfull[k]
        den += float((G ** 2).sum())
        sub = method.subspace(k)
        if sub is None:
            continue
        has_sub = True
        U, V = sub
        UtG = U.T @ G
        GV = G @ V
        P = U @ UtG + GV @ V.T - U @ (UtG @ V) @ V.T
        num += float((P ** 2).sum())
    if not has_sub:
        return float("nan")
    return (num / max(den, 1e-30)) ** 0.5


# ── Shift reference: coverage of G_t over the FROZEN top-r(G_0) ────────────────────
# Shows whether the dominant rank-r directions of the full-weight gradient move from
# t=0 to t -- a statement about the optimization landscape, built from Gfull itself
# (the raw weight gradient). Used as the coverage reference for any method that exposes
# no subspace of its own.

def _topr(G, r):
    """Top-r left/right singular subspace of one gradient matrix (cheap, truncated)."""
    G = G.detach().float()
    r = min(int(r), min(G.shape))
    q = min(min(G.shape), r + 6)                       # small oversampling for accuracy
    U, S, V = torch.svd_lowrank(G, q=q)                # G ≈ U diag(S) Vᵀ
    return U[:, :r].contiguous(), V[:, :r].contiguous()


def _coverage_ref(Gfull, keys, sub0) -> float:
    num = den = 0.0
    for k in keys:
        G = Gfull[k].detach().float()
        den += float((G ** 2).sum())
        uv = sub0.get(k)
        if uv is None:
            continue
        U, V = uv
        UtG = U.t() @ G
        GV = G @ V
        P = U @ UtG + GV @ V.t() - U @ (UtG @ V) @ V.t()
        num += float((P ** 2).sum())
    return (num / max(den, 1e-30)) ** 0.5


def _principal_angles(subA, subB, keys, side):
    per_layer = []
    for k in keys:
        a, b = subA.get(k), subB.get(k)
        if a is None or b is None:
            continue
        A, B = a[side], b[side]
        m = min(A.shape[1], B.shape[1])
        s = torch.linalg.svdvals(A[:, :m].t() @ B[:, :m]).clamp(-1.0, 1.0)
        per_layer.append(torch.arccos(s)[:m])
    if not per_layer:
        return None
    m = min(t.numel() for t in per_layer)
    return torch.stack([t[:m] for t in per_layer]).mean(dim=0).tolist()


def _infer_rank(method, keys, default=8) -> int:
    try:
        sub = method.subspace(keys[0])
        if sub is not None:
            return int(sub[0].shape[1])
    except Exception:
        pass
    spec = getattr(method, "spec", None)
    for attr in ("r", "rank"):                 # MethodSpec stores rank as `r`
        v = getattr(spec, attr, None)
        if v:
            return int(v)
    return default


def _snapshot(target_mods):
    return {k: m.weight.detach().float().cpu().clone() for k, m in target_mods.items()}


def _restore(target_mods, snap):
    for k, m in target_mods.items():
        set_weight(m, snap[k])


def train_lm(model, target_mods, refs, method, train_batches, lr, steps,
             schedule=None, checkpoint_fn=None,
             eval_steps=None, early_stop=False, patience=3, monitor="val_ce",
             mode="min", log=None, probe_fn=None, probe_every=0,
             coverage_pre_step=True, shift_angles=True, rotation_window=0,
             save_subspaces=False, save_dir=None, restore_best=True, seed=0):
    import time
    keys = list(refs)
    sched = schedule or (lambda t: 1.0)
    wd = float(getattr(method.spec, "adam_weight_decay", 0.0))
    eval_at = set(eval_steps or [])
    # bg's per-sample evidence draws (_micro_batches) are the ONLY stochastic part of
    # the bg_lora step, so they must ride the run's seed -- otherwise every seed shares
    # the exact same evidence sequence. Derive it from cfg.seed so ONE seed governs the
    # run's randomness.
    gen = torch.Generator().manual_seed(int(seed))     # seed-dependent evidence draws
    dev = next(model.parameters()).device              # run the filter math here (GPU)
    method.compute_device = dev

    if method.needs_init_gradient:
        G0, _ = weight_grads(model, target_mods, train_batches[0])
    else:
        G0 = {k: None for k in keys}

    for k in keys:
        W0 = target_mods[k].weight.detach().float().clone()         # on the model device
        set_weight(target_mods[k], method.init_layer(refs[k], W0, G0[k]))

    better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
    best_val = float("inf") if mode == "min" else float("-inf")
    best_snap = _snapshot(target_mods) if restore_best else None
    best_step, stale = -1, 0

    cuda = torch.cuda.is_available()
    R = method.refresh_interval                        # bg: rotate every R steps
    # ── Coverage reference ────────────────────────────────────────────────────
    # ONE `coverage` number per step: the fraction of the FULL gradient G_t captured
    # by bg_lora's rotating rank-r subspace. Staying high = the subspace tracks the
    # gradient; falling would mean it drifts out of alignment. (For a method that
    # exposes no subspace, the reference is instead the FROZEN top-r(G_0), which decays
    # as the gradient's own subspace shifts from t=0.)
    sub0 = None
    sub_win = None                                     # window reference for rot_* angles
    ref_r = _infer_rank(method, list(refs))
    train_rows, eval_rows = [], []
    tokens_seen, train_wall, train_peak = 0, 0.0, 0
    probe_wall, refresh_wall, refresh_n = 0.0, 0.0, 0
    diverged = False
    for t in range(steps):
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        method.cur_step = t                    # global step index; bg_lora tags raw z-evidence with it
        s0 = time.perf_counter()
        batch = train_batches[t % len(train_batches)]
        tokens_seen += int(batch["attention_mask"].sum())
        is_refresh = (t % R == 0)
        need_ps = method.needs_per_sample and is_refresh   # per-sample only on refresh
        Gfull, loss = weight_grads(model, target_mods, batch)
        # DIVERGENCE GUARD. If the loss is already non-finite, the weights blew up on a
        # previous step and Gfull is NaN/Inf. Stop THIS run cleanly (keep best weights,
        # flag it) instead of feeding it into eigh/svd and killing the container.
        if not np.isfinite(loss):
            if log:
                print(f"  {method.name:16s} t={t:3d} NON-FINITE loss -> stop "
                      f"(diverged); keeping best {monitor}={best_val:.3f}@{best_step}")
            diverged = True
            break
        # Freeze the t=0 full-gradient top-r subspace once (the SHIFT reference),
        # reusing the Gfull we just computed.
        if sub0 is None:
            sub0 = {k: _topr(Gfull[k], ref_r) for k in keys}
        _sp0 = time.perf_counter()
        sub_pre = {k: method.subspace(k) for k in keys} if coverage_pre_step else None
        snap_dt = time.perf_counter() - _sp0
        Gps = (per_sample_weight_grads(
                   model, target_mods,
                   _micro_batches(batch, method.spec.evidence_n,
                                  method.spec.evidence_bs, gen))
               if need_ps else None)
        lr_t = lr * sched(t)                     # real LR, shared warmup+cosine schedule
        grad_norm = float(sum(float(Gfull[k].norm()) for k in keys))
        upd, wnorm = 0.0, 0.0
        for k in keys:
            evidence = Gps[k] if need_ps else Gfull[k]
            step = method.step(refs[k], evidence, is_refresh)
            W_cur = target_mods[k].weight.detach().float()           # on model device
            # The proposal is already Adam-preconditioned; apply the real LR directly.
            dW = step.commit(lr_t)
            if wd:                                                   # decoupled weight decay
                dW = dW - lr_t * wd * W_cur
            if not bool(torch.isfinite(dW).all()):     # catch the diverging step at its
                diverged = True                        # source, before it poisons W
                break
            add_weight(target_mods[k], dW)
            upd += float(dW.norm())
            wnorm += float((W_cur + dW).norm())
        if diverged:
            if log:
                print(f"  {method.name:16s} t={t:3d} NON-FINITE update -> stop "
                      f"(diverged); keeping best {monitor}={best_val:.3f}@{best_step}")
            break
        if cuda:
            torch.cuda.synchronize()
            train_peak = max(train_peak, int(torch.cuda.max_memory_allocated() / (1024 ** 2)))
        step_dt = time.perf_counter() - s0
        train_wall += step_dt - snap_dt        # keep the pre-step snapshot out of train timing
        probe_wall += snap_dt
        # ONE coverage number: fraction of the full gradient captured by the method's
        # rank-r subspace (or, if it exposes none, by the frozen top-r(G_0)).
        sub_use = sub_pre if coverage_pre_step else {k: method.subspace(k) for k in keys}
        has_sub = any(v is not None for v in sub_use.values())
        if has_sub:
            cov = _coverage_ref(Gfull, keys, sub_use)
        else:
            cov = _coverage_ref(Gfull, keys, sub0)
        row = {"step": t, "train_loss": loss, "learning_rate": lr_t,
               "grad_norm": grad_norm, "update_norm": upd, "weight_norm": wnorm,
               "coverage": cov,
               "is_refresh": bool(is_refresh),
               "tokens_seen": tokens_seen, "step_seconds": round(step_dt - snap_dt, 4),
               "train_seconds": round(train_wall, 3)}
        # Subspace principal angles of the FULL gradient's top-r geometry, timed into
        # probe_wall (train_seconds unaffected). cur = top-r(Gfull) is computed once and
        # reused: pa_i = U_t vs U_0 (left shift from t=0), pa_v_i = V_t vs V_0; and every
        # `rotation_window` steps (0 -> R) the CONSECUTIVE-window rotation rot_u_i /
        # rot_v_i (how fast the subspace is turning NOW).
        if shift_angles and sub0 is not None:
            _pa0 = time.perf_counter()
            cur = {k: _topr(Gfull[k], ref_r) for k in keys}
            au = _principal_angles(cur, sub0, keys, 0)          # U_t vs U_0
            av = _principal_angles(cur, sub0, keys, 1)          # V_t vs V_0
            if au:
                for i, a in enumerate(au, 1): row[f"pa_{i}"] = float(a)
            if av:
                for i, a in enumerate(av, 1): row[f"pa_v_{i}"] = float(a)
            win = rotation_window or R
            if sub_win is None:
                sub_win = cur                                   # seed the window ref at t=0
            elif t % win == 0:                                  # boundary: t = win, 2*win, ...
                ru = _principal_angles(cur, sub_win, keys, 0)   # U_t vs U_{t-window}
                rv = _principal_angles(cur, sub_win, keys, 1)   # V_t vs V_{t-window}
                if ru:
                    for i, a in enumerate(ru, 1): row[f"rot_u_{i}"] = float(a)
                if rv:
                    for i, a in enumerate(rv, 1): row[f"rot_v_{i}"] = float(a)
                sub_win = cur                                   # advance the window reference
            probe_wall += time.perf_counter() - _pa0
        # Raw subspace snapshots (full arm): top-r (U,S,V) of the FULL gradient at the
        # t=0,window,2*window,... cadence, for offline geometry figures. Timed into probe_wall.
        if save_subspaces and save_dir is not None and (t % (rotation_window or R) == 0):
            _sv0 = time.perf_counter()
            _save_subspaces(save_dir, t, Gfull, keys, ref_r)
            probe_wall += time.perf_counter() - _sv0
        train_rows.append(row)
        # Refresh steps are the expensive ones for bg (evidence_n extra backward passes +
        # the Grassmann map); timing them apart makes the amortised refresh cost measurable.
        if is_refresh:
            refresh_wall += step_dt
            refresh_n += 1

        if probe_fn is not None and probe_every and (t % probe_every == 0):
            p0 = time.perf_counter()
            try:
                probe_fn(t, method, refs, keys, Gfull)
            except Exception as e:                 # a diagnostic must never kill a run
                if log:
                    print(f"  {method.name:16s} t={t:3d} probe failed: "
                          f"{type(e).__name__}: {e}")
            probe_wall += time.perf_counter() - p0

        is_ckpt = checkpoint_fn is not None and (t in eval_at or t == steps - 1)
        if is_ckpt:
            ctx = {"train_seconds": train_wall, "train_peak_gpu_mib": train_peak,
                   "tokens_seen": tokens_seen}
            metrics = checkpoint_fn(t, method, refs, keys, Gfull, ctx)
            eval_rows.append({"step": t, **metrics})
            val = metrics.get(monitor)
            if val is not None and better(val, best_val):
                best_val, best_step, stale = val, t, 0
                if restore_best:
                    best_snap = _snapshot(target_mods)
            else:
                stale += 1
            if log:
                print(f"  {method.name:16s} t={t:3d} loss={loss:.4f} "
                      f"cov={row['coverage']:.3f} " +
                      " ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                               if isinstance(v, (int, float))) +
                      f"  (best {monitor}={best_val:.3f}@{best_step}, stale={stale})")
            if early_stop and stale >= patience:
                if log:
                    print(f"  {method.name:16s} early stop at t={t}")
                break
        elif log and (t % max(log, 1) == 0):
            print(f"  {method.name:16s} t={t:3d} loss={loss:.4f} lr={lr_t:.2e} "
                  f"cov={row['coverage']:.3f}")

    if restore_best and best_snap is not None:
        _restore(target_mods, best_snap)            # GLUE: compare near best, not endpoint
    # else: keep the LAST-step weights (generative fixed-budget protocol)
    n_steps = max(len(train_rows), 1)
    n_frozen = max(n_steps - refresh_n, 0)
    out = {"best_step": best_step, "best_val": best_val,
           "monitor": monitor, "diverged": diverged,
           "probe_seconds": round(probe_wall, 3)}
    # The refresh/frozen split only MEANS anything for a method that does something
    # different on a refresh step (i.e. consumes per-sample evidence).
    if getattr(method, "needs_per_sample", False) and n_frozen > 0:
        out.update({
            "refresh_steps": refresh_n, "frozen_steps": n_frozen,
            "refresh_seconds": round(refresh_wall, 3),
            "frozen_seconds": round(train_wall - refresh_wall, 3),
            "refresh_seconds_mean": round(refresh_wall / max(refresh_n, 1), 4),
            "frozen_seconds_mean": round((train_wall - refresh_wall) / n_frozen, 4),
        })
    else:
        out.update({"refresh_steps": None, "frozen_steps": None,
                    "refresh_seconds": None, "frozen_seconds": None,
                    "refresh_seconds_mean": None, "frozen_seconds_mean": None})
    return train_rows, eval_rows, out