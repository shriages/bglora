#!/usr/bin/env python3
"""Per-DIRECTION local rotation, U and V kept separate (not merged over the r atoms).

Reads the consecutive-window principal angles from train_metrics.jsonl:
    rot_u_1 .. rot_u_r   left  (U) subspace, angle between U_t and U_{t-window}, per index
    rot_v_1 .. rot_v_r   right (V) subspace, same
(index i is the i-th principal angle, sorted small->large; i=1 is the best-aligned pair,
i=r the least. These are the r "directions" of the subspace rotation.) Logged every
rotation_window steps (default = refresh_interval).

Two views:
  profile (default): x = direction index 1..r, y = mean local rotation over training,
      one line per run, shaded IQR. U panel | V panel. The crisp "which directions turn,
      and does one task turn more" comparison.
  timeseries: a grid (rows = runs, cols = U/V), each cell showing all r directions as
      lines over training. The detailed view.

Local turn-rate, so runs at different LR are directly comparable.

Usage
-----
  python plot_dir_rotation.py --run 'runs/gsm8k/*/bg_lora_adam/seed_11*:gsm8k' \
                              --run 'runs/math/*/bg_lora_adam/seed_3*:MATH'
  python plot_dir_rotation.py --run ... --run ... --mode timeseries
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _find(path, fname="train_metrics.jsonl"):
    if any(c in path for c in "*?["):
        cands = (glob.glob(os.path.join(path, fname))
                 + glob.glob(os.path.join(path, "**", fname), recursive=True)
                 + [p for p in glob.glob(path, recursive=True) if p.endswith(fname)])
        cands = sorted({p for p in cands if os.path.exists(p)})
        if len(cands) > 1:
            print(f"[info] {path!r} matched {len(cands)}; using {cands[0]}")
        return cands[0] if cands else None
    if os.path.isdir(path):
        p = os.path.join(path, fname)
        return p if os.path.exists(p) else None
    return path if os.path.exists(path) and path.endswith(fname) else None


def _rows(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _collect(path, side):
    """Return (steps, matrix[n_steps, r]) of rot_{side}_i angles, only rows that have them."""
    import numpy as np
    p = _find(path)
    if not p:
        return None
    pref = f"rot_{side}_"
    steps, mat = [], []
    for r in _rows(p):
        idx = sorted(int(k[len(pref):]) for k in r if k.startswith(pref) and k[len(pref):].isdigit())
        if not idx:
            continue
        row = [r.get(f"{pref}{i}", float("nan")) for i in idx]
        if any(v == v for v in row):                 # at least one finite
            steps.append(r.get("step", len(steps)))
            mat.append(row)
    if not steps:
        return None
    R = max(len(m) for m in mat)
    M = np.full((len(mat), R), np.nan)
    for i, m in enumerate(mat):
        M[i, :len(m)] = m
    return np.array(steps, float), M


def _cov_series(path):
    """(steps, coverage) from train_metrics.jsonl, finite rows only."""
    import numpy as np
    p = _find(path)
    if not p:
        return None
    steps, cov = [], []
    for r in _rows(p):
        c = r.get("coverage")
        if isinstance(c, (int, float)) and c == c:
            steps.append(r.get("step", len(steps)))
            cov.append(float(c))
    if not steps:
        return None
    return np.array(steps, float), np.array(cov, float)


def _smooth(y, w):
    """Centered rolling mean, window w (steps ~ samples); NaN-safe, keeps length."""
    import numpy as np
    w = int(w)
    if w <= 1 or y.size < 3:
        return y
    w = min(w, y.size if y.size % 2 else y.size - 1)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return y
    pad = w // 2
    yp = np.pad(y, pad, mode="edge")
    ker = np.ones(w) / w
    return np.convolve(yp, ker, mode="valid")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    help="run dir / file for the U/V ROTATION panels (use full-FT ground-truth "
                         "runs), optionally ':label'; glob ok; repeatable")
    ap.add_argument("--cov-run", dest="cov_run", action="append", default=None,
                    help="run dir / file for the COVERAGE panel (use YOUR bg_lora runs -- full-FT "
                         "has no coverage), optionally ':label'; glob ok; repeatable. Adds a 3rd panel.")
    ap.add_argument("--smooth", type=int, default=51,
                    help="coverage smoothing window in steps (centered rolling mean; 1 = off).")
    ap.add_argument("--mode", default="profile", choices=["profile", "timeseries"])
    ap.add_argument("--out", default="dir_rotation.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    runs = []
    for spec in args.run:
        path, _, label = spec.partition(":")
        u = _collect(path, "u")
        v = _collect(path, "v")
        if u is None and v is None:
            print(f"[warn] no rot_u_*/rot_v_* in {path!r} -- skipping "
                  f"(was shift_angles on? train_metrics needs rot_* rows)")
            continue
        runs.append((label or os.path.basename(path), u, v))
    if not runs:
        raise SystemExit("no usable runs with per-direction rot_u_*/rot_v_* angles")

    cmap = plt.get_cmap("tab10")

    # coverage-panel runs (bg_lora): read here so we know whether to make a 3rd panel
    cov_runs = []
    for spec in (args.cov_run or []):
        path, _, label = spec.partition(":")
        s = _cov_series(path)
        if s is None:
            print(f"[warn] no coverage in {path!r} -- skipping (full-FT has none; use bg runs)")
            continue
        cov_runs.append((label or os.path.basename(path), *s))

    if args.mode == "profile":
        ncols = 3 if cov_runs else 2
        fig, axes = plt.subplots(1, ncols, figsize=(6.4 * ncols, 5))
        axU, axV = axes[0], axes[1]
        for si, (label, u, v) in enumerate(runs):
            c = cmap(si % 10)
            for ax, data, name in ((axU, u, "U"), (axV, v, "V")):
                if data is None:
                    continue
                _, M = data
                r = M.shape[1]
                x = np.arange(1, r + 1)
                med = np.nanmedian(M, axis=0)
                q1 = np.nanpercentile(M, 25, axis=0)
                q3 = np.nanpercentile(M, 75, axis=0)
                ax.plot(x, med, "-o", color=c, lw=1.8, ms=5, label=label)
                ax.fill_between(x, q1, q3, color=c, alpha=0.18)
        axV.sharey(axU)
        for ax, name in ((axU, "U (left) subspace"), (axV, "V (right) subspace")):
            ax.set_title(name); ax.set_xlabel("direction index (principal angle 1..r)")
            ax.grid(True, alpha=0.25)
        axU.set_ylabel("local rotation angle (rad), median over training")
        axU.legend()
        # ---- 3rd panel: coverage over training (bg method), smoothed, colored by task ----
        if cov_runs:
            axC = axes[2]
            for ci, (label, steps, cov) in enumerate(cov_runs):
                c = cmap(ci % 10)
                order = np.argsort(steps)
                st, cv = steps[order], cov[order]
                axC.plot(st, cv, "-", color=c, lw=0.8, alpha=0.22)          # raw (faint)
                axC.plot(st, _smooth(cv, args.smooth), "-", color=c, lw=2.2, label=label)  # smoothed
                print(f"  coverage {label:20s} start={cv[0]:.3f} end={cv[-1]:.3f} "
                      f"mean={np.mean(cv):.3f}")
            axC.set_title(f"coverage over training (bg; smooth={args.smooth})")
            axC.set_xlabel("training step")
            axC.set_ylabel("coverage (grad energy in rank-r U,V subspace)")
            axC.set_ylim(0, 1); axC.grid(True, alpha=0.25); axC.legend()
        fig.suptitle("Per-direction local rotation (median ± IQR)"
                     + ("  +  method coverage" if cov_runs else ""), y=1.02)
        # console table
        print(f"\n{'dir':>3} | " + " | ".join(f"{lab[:10]:>10}(U/V)" for lab, _, _ in runs))
        rmax = max((u[1].shape[1] if u else 0) for _, u, v in runs)
        for i in range(rmax):
            cells = []
            for _, u, v in runs:
                uu = f"{np.nanmedian(u[1][:, i]):.3f}" if (u and i < u[1].shape[1]) else "  -  "
                vv = f"{np.nanmedian(v[1][:, i]):.3f}" if (v and i < v[1].shape[1]) else "  -  "
                cells.append(f"{uu}/{vv}")
            print(f"{i+1:>3} | " + " | ".join(f"{c:>16}" for c in cells))
    else:  # timeseries grid: rows = runs, cols = (U, V)
        nr = len(runs)
        fig, axes = plt.subplots(nr, 2, figsize=(13, 3.4 * nr), squeeze=False)
        for si, (label, u, v) in enumerate(runs):
            for ci, (data, name) in enumerate(((u, "U"), (v, "V"))):
                ax = axes[si][ci]
                if data is not None:
                    steps, M = data
                    r = M.shape[1]
                    dc = plt.get_cmap("viridis")
                    for k in range(r):
                        ax.plot(steps, M[:, k], "-", lw=1.2, color=dc(k / max(r - 1, 1)),
                                label=f"dir {k+1}")
                    if si == 0 and ci == 1:
                        ax.legend(fontsize=6, ncol=2, loc="upper right")
                ax.set_title(f"{label} — {name}")
                ax.set_xlabel("training step"); ax.grid(True, alpha=0.25)
                if ci == 0:
                    ax.set_ylabel("local rotation (rad)")
        fig.suptitle("Per-direction local rotation over training (color = direction 1..r)", y=1.0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("\nwrote", args.out)


if __name__ == "__main__":
    raise SystemExit(main())