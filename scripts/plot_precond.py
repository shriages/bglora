#!/usr/bin/env python3
"""The naive preconditioner story: per-direction GAIN vs rotation angle.

For each atom k at each refresh, bg_lora (with GRABLORA_PRECOND_TRACE=1) now logs BOTH
denominators at the SAME rotation angle:

    theta_V     angle between A_{t,k} and A_{t-1,k}   (right-subspace principal angle, rad)
    gain        1/(sqrt(obs_var + P_pred) + eps)      <-- OURS  (rotation-aware)
    gain_adam   1/(sqrt(v_hat) + eps)                 <-- plain Adam (rotation-blind)

Both come from ONE run, so "same angle" is exact -- it is literally the same row.

The figure:  x = theta_V,  y = gain.  Two series.
  * OURS bends DOWN as the angle grows  (bigger rotation -> bigger P_pred -> bigger
    denominator -> smaller step).
  * plain Adam is FLAT in the angle       (v_hat never sees the rotation).

That gap, read at a fixed angle, is exactly "how our edited-Adam step size differs from
plain Adam's for the same rotation".

Usage
-----
  python plot_gain_vs_angle.py --trace .../precond_trace.jsonl
  python plot_gain_vs_angle.py --trace .../precond_trace.jsonl --logy --out gain.png
  python plot_gain_vs_angle.py --trace .../precond_trace.jsonl --layer q_proj --step-max 200
"""
from __future__ import annotations

import argparse
import json


def load_trace(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="precond_trace.jsonl from a bg_lora run")
    ap.add_argument("--out", default="gain_vs_angle.png")
    ap.add_argument("--phase", default="refresh", choices=["refresh", "init", "all"])
    ap.add_argument("--layer", default="", help="keep only rows whose layer contains this substring")
    ap.add_argument("--step-min", type=float, default=None, help="drop rows with step < this")
    ap.add_argument("--step-max", type=float, default=None, help="drop rows with step > this")
    ap.add_argument("--logy", action="store_true", help="log scale on the gain axis")
    ap.add_argument("--bins", type=int, default=12, help="angle bins for the median trend line")
    ap.add_argument("--max-points", type=int, default=8000, help="subsample scatter for legibility")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = load_trace(args.trace)
    if args.phase != "all":
        rows = [r for r in rows if r.get("phase") == args.phase]
    if args.layer:
        rows = [r for r in rows if args.layer in str(r.get("layer", ""))]
    if args.step_min is not None:
        rows = [r for r in rows if r.get("step") is not None and r["step"] >= args.step_min]
    if args.step_max is not None:
        rows = [r for r in rows if r.get("step") is not None and r["step"] <= args.step_max]
    rows = [r for r in rows if r.get("theta_V") is not None and r.get("gain") is not None]
    if not rows:
        raise SystemExit("no usable rows after filtering (need theta_V and gain; "
                         "was the run bg_lora with coeff_optimizer=adam and PRECOND_TRACE=1?)")
    if "gain_adam" not in rows[0]:
        raise SystemExit("trace has no 'gain_adam' -- rerun with the updated bg_lora.py "
                         "so the plain-Adam comparison denominator is logged.")

    def arr(key):
        return np.array([r.get(key, np.nan) for r in rows], dtype=float)

    theta = arr("theta_V")
    g_ours = arr("gain")
    g_adam = arr("gain_adam")
    good = np.isfinite(theta) & np.isfinite(g_ours) & np.isfinite(g_adam)
    theta, g_ours, g_adam = theta[good], g_ours[good], g_adam[good]

    # subsample scatter only (trend line uses all points)
    if len(theta) > args.max_points:
        idx = np.random.default_rng(0).choice(len(theta), args.max_points, replace=False)
        ts, gos, gas = theta[idx], g_ours[idx], g_adam[idx]
    else:
        ts, gos, gas = theta, g_ours, g_adam

    C_OURS, C_ADAM = "#1f77b4", "#d62728"

    def binned_median(x, y, nb):
        if x.max() <= x.min():
            return np.array([x.mean()]), np.array([np.median(y)])
        edges = np.linspace(x.min(), x.max(), nb + 1)
        cx, cy = [], []
        for i in range(nb):
            sel = (x >= edges[i]) & (x <= edges[i + 1] if i == nb - 1 else x < edges[i + 1])
            if sel.sum() >= 3:
                cx.append(0.5 * (edges[i] + edges[i + 1]))
                cy.append(np.median(y[sel]))
        return np.array(cx), np.array(cy)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(ts, gos, s=9, alpha=0.30, color=C_OURS, edgecolors="none", label="GRAB-LoRA (ours)")
    ax.scatter(ts, gas, s=9, alpha=0.30, color=C_ADAM, edgecolors="none", label="plain Adam")
    for y, c in ((g_ours, C_OURS), (g_adam, C_ADAM)):
        bx, by = binned_median(theta, y, args.bins)
        if len(bx) > 1:
            ax.plot(bx, by, "-", color=c, lw=2.2, zorder=5)

    ax.set_xlabel(r"$\theta_{V,k}$  = angle$(A_{t,k},\,A_{t-1,k})$   (rad)")
    ax.set_ylabel(r"gain  $=\ 1/(\sqrt{\mathrm{var}}+\epsilon)$   (per-direction step multiplier)")
    if args.logy:
        ax.set_yscale("log")
    ttl = "Preconditioner gain vs rotation angle"
    if args.step_min is not None or args.step_max is not None:
        lo = "" if args.step_min is None else f"{args.step_min:g}"
        hi = "" if args.step_max is None else f"{args.step_max:g}"
        ttl += f"   (steps {lo}–{hi})"
    ax.set_title(ttl)
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", args.out)

    # quantify the trend: Spearman rho(theta, gain) per series (ours should be strongly
    # negative = rotation damps the step; plain Adam should be ~0 = rotation-blind).
    def spearman(x, y):
        if len(x) < 3 or x.std() == 0 or y.std() == 0:
            return float("nan")
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    print(f"  ours   : spearman(theta, gain)      = {spearman(theta, g_ours):+.3f}   "
          f"(negative = rotation damps the step)")
    print(f"  adam   : spearman(theta, gain_adam) = {spearman(theta, g_adam):+.3f}   "
          f"(~0 = rotation-blind)")
    print(f"  median gain  ours/adam = {np.median(g_ours):.4g} / {np.median(g_adam):.4g}")


if __name__ == "__main__":
    raise SystemExit(main())