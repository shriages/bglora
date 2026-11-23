#!/usr/bin/env python3
"""Per-layer rotation grids: {q_proj, v_proj} x {local, global}, U | V panels.

Reads the SVD snapshot tree the `full` arm writes (lm_loop._save_subspaces):

    <RUN_DIR>/subspaces/
        meta.json
        <layer>/t000000.npz, t000020.npz, ...     # arrays: U, S, V, step

and produces FOUR figures = {q_proj, v_proj} x {local, global}. In each figure:

    left  half, captioned "U" : a 7x2 grid of that module's layers, U-subspace
    right half, captioned "V" : the same layers, V-subspace

    local  = incremental HEATMAP: angle(sub_{t-tau}, sub_t)   (G_t vs G_{t-tau})
             x = training step, y = principal-angle index k, color = deg.
    global = cumulative LINE plot: angle(sub_0, sub_t)        (G_t vs G_0)
             x = training step, y = principal angle (deg), one line per index k.

MODEL-AGNOSTIC LAYER PICK (new): when --layers-list is NOT given, the 14 grid
panels are chosen as 14 EVENLY-SPACED layers from whatever the run actually
contains (first and last always included). So this works for Llama-3.1-8B
(32 layers, 0..31), Qwen2.5-1.5B (28), T5 (24), etc. with no flag. Override any
time with --layers-list.

NOTE: the snapshots only exist for a run trained with method == "full" (the arm
that saves them). So point --run_dir at a full-FT run's directory, e.g.
runs/gsm8k/<llama-slug>/full/seed_<s>.

"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

HALF_PI = np.pi / 2.0

# Fallback default run dir (override on the command line). Point at a full-FT run.
DEFAULT_RUN_DIR = (
    "runs/gsm8k/Llama-3.1-8B/full/seed_1"
)

# When --layers-list is omitted we AUTO-pick this many, evenly spaced, from the
# layers present in the run (see _auto_layers). Kept as a constant so the grid
# shape stays 7 x 2.
NROWS, NCOLS = 7, 2
CMAP = "magma"

KLINE = [
    ("#1B9E77", "-"), ("#D95F02", "-"), ("#7570B3", "-"), ("#E7298A", "-"),
    ("#66A61E", "-"), ("#E6AB02", "-"), ("#A6761D", "-"), ("#666666", "-"),
]


def _kline(k):
    return KLINE[k % len(KLINE)]


# ─────────────────────────────────────────────────────────────────────────────
# Reading + geometry
# ─────────────────────────────────────────────────────────────────────────────
def load_run(run_dir, subspaces_dir=None):
    base = subspaces_dir or os.path.join(run_dir, "subspaces")
    meta_path = os.path.join(base, "meta.json")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"no subspaces directory at: {base}\n"
                                f"  (only a method=='full' run writes these)")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"no meta.json at: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    data = {}
    for layer in list(meta["layers"]):
        ldir = os.path.join(base, layer.replace("/", "_"))
        files = sorted(glob.glob(os.path.join(ldir, "t*.npz")))
        if not files:
            continue
        Us, Ss, Vs, ts = [], [], [], []
        for fpath in files:
            z = np.load(fpath)
            Us.append(np.asarray(z["U"], dtype=np.float64))
            Ss.append(np.asarray(z["S"], dtype=np.float64))
            Vs.append(np.asarray(z["V"], dtype=np.float64))
            ts.append(int(z["step"]))
        order = np.argsort(ts)
        data[layer] = {"steps": [ts[i] for i in order],
                       "U": [Us[i] for i in order], "S": [Ss[i] for i in order],
                       "V": [Vs[i] for i in order]}
    if not data:
        raise RuntimeError(f"found meta.json but no .npz snapshots under {base}")
    step_sets = [set(d["steps"]) for d in data.values()]
    common = sorted(set.intersection(*step_sets)) if step_sets else []
    if not common:
        raise RuntimeError("no common steps across layers")
    return meta, data, common


def _index_maps(data, steps):
    return {layer: {s: i for i, s in enumerate(d["steps"])}
            for layer, d in data.items()}


def principal_angles(A, B):
    m = min(A.shape[1], B.shape[1])
    s = np.linalg.svd(A[:, :m].T @ B[:, :m], compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def local_angles_one_layer(layer_data, imap_layer, steps, side):
    cols, end_steps = [], []
    for i in range(1, len(steps)):
        s_prev, s_cur = steps[i - 1], steps[i]
        p0, p1 = imap_layer.get(s_prev), imap_layer.get(s_cur)
        cols.append(None if p0 is None or p1 is None
                    else principal_angles(layer_data[side][p0], layer_data[side][p1]))
        end_steps.append(s_cur)
    r = max((len(c) for c in cols if c is not None), default=0)
    M = np.full((r, len(cols)), np.nan)
    for j, c in enumerate(cols):
        if c is not None:
            M[: len(c), j] = c
    return M, end_steps


def global_lines_one_layer(layer_data, imap_layer, steps, side):
    p0 = imap_layer.get(steps[0])
    rows, used = [], []
    for s in steps:
        p = imap_layer.get(s)
        if p0 is None or p is None:
            continue
        rows.append(principal_angles(layer_data[side][p0], layer_data[side][p]))
        used.append(s)
    if not rows:
        return None, None
    r = max(len(x) for x in rows)
    A = np.full((len(rows), r), np.nan)
    for i, x in enumerate(rows):
        A[i, : len(x)] = x
    return np.asarray(used, dtype=float), A


def _layer_index(key):
    m = re.search(r"layers?\.(\d+)", key) or re.search(r"block[._](\d+)", key)
    return int(m.group(1)) if m else None


def _module(key):
    m = re.search(r"(q_proj|v_proj)", key)
    if m:
        return m.group(1)
    # T5 fallback: SelfAttention .q / .v
    m = re.search(r"\.(q|v)$", key) or re.search(r"\.(q|v)\.", key)
    if m:
        return {"q": "q_proj", "v": "v_proj"}[m.group(1)]
    return None


def build_lookup(data):
    lut = {}
    for key in data:
        i, mod = _layer_index(key), _module(key)
        if i is not None and mod is not None:
            lut[(i, mod)] = key
    return lut


def _auto_layers(data, n):
    """Pick n evenly-spaced layer indices from those present (first & last always
    included). Model-agnostic: 32-layer Llama, 28-layer Qwen, 24-block T5, ..."""
    idxs = sorted({_layer_index(k) for k in data if _layer_index(k) is not None})
    if not idxs:
        raise SystemExit("could not parse any layer indices from snapshot keys")
    if len(idxs) < n:
        raise SystemExit(f"run has only {len(idxs)} layers with q/v snapshots; "
                         f"need {n} for a {NROWS}x{NCOLS} grid. Pass --layers-list "
                         f"with exactly {n} of: {idxs}")
    sel = [idxs[int(round(i * (len(idxs) - 1) / (n - 1)))] for i in range(n)]
    # dedupe while preserving even spread (shouldn't trigger when len>=n distinct)
    seen, out = set(), []
    for x in sel:
        if x not in seen:
            seen.add(x); out.append(x)
    j = 0
    while len(out) < n and j < len(idxs):
        if idxs[j] not in seen:
            seen.add(idxs[j]); out.append(idxs[j])
        j += 1
    return sorted(out)[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=DEFAULT_RUN_DIR,
                    help="run directory containing subspaces/ (a method=='full' run)")
    ap.add_argument("--subspaces-dir", default=None,
                    help="path straight to a subspaces/ folder (overrides run_dir)")
    ap.add_argument("--out", default="layer_grid.png",
                    help="output stem; module and rotation-type tags are appended")
    ap.add_argument("--layers-list", default=None,
                    help=f"comma-separated layer indices for the {NROWS*NCOLS} panels "
                         f"(default: auto — 14 evenly-spaced from the run's own layers)")
    ap.add_argument("--vmax", type=float, default=None,
                    help="max in DEGREES: colorbar cap (local) / y-limit (global). "
                         "Default = data max per rotation-type; use 90 for [0,pi/2].")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["font.weight"] = "bold"
    matplotlib.rcParams["axes.labelweight"] = "bold"
    matplotlib.rcParams["axes.titleweight"] = "bold"
    matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    base = args.subspaces_dir or os.path.join(args.run_dir, "subspaces")
    print(f"Loading snapshots from: {base}")
    meta, data, steps = load_run(args.run_dir, args.subspaces_dir)
    imap = _index_maps(data, steps)
    lut = build_lookup(data)

    layers = ([int(x) for x in args.layers_list.split(",")]
              if args.layers_list else _auto_layers(data, NROWS * NCOLS))
    if len(layers) != NROWS * NCOLS:
        raise SystemExit(f"--layers-list has {len(layers)} entries; need "
                         f"{NROWS * NCOLS} for a {NROWS}x{NCOLS} grid")
    print(f"  layers loaded : {len(data)}   steps: {len(steps)} "
          f"({steps[0]}..{steps[-1]})   rank r: {meta.get('rank')}")
    print(f"  panel layers  : {layers}")

    modules = [("q_proj", "q"), ("v_proj", "v")]
    rottypes = [("local", "local"), ("global", "global")]
    caps = [("U", "U"), ("V", "V")]

    allpanels = {}
    gmax = {"local": 0.0, "global": 0.0}
    for mode, _ in rottypes:
        for module, _ in modules:
            per_cap = {}
            for caption, side in caps:
                plist = []
                for li in layers:
                    key = lut.get((li, module))
                    if key is None:
                        plist.append((li, None, None)); continue
                    if mode == "local":
                        M, xs = local_angles_one_layer(data[key], imap[key], steps, side)
                        payload = np.degrees(M)
                    else:
                        xs, A = global_lines_one_layer(data[key], imap[key], steps, side)
                        payload = np.degrees(A) if A is not None else None
                    if payload is not None and np.isfinite(payload).any():
                        gmax[mode] = max(gmax[mode], float(np.nanmax(payload)))
                    plist.append((li, payload, xs))
                per_cap[caption] = plist
            allpanels[(mode, module)] = per_cap

    scale = {}
    for mode in ("local", "global"):
        scale[mode] = args.vmax if args.vmax is not None else (gmax[mode] or 90.0)
    norm_local = Normalize(vmin=0.0, vmax=scale["local"])
    ytop_global = scale["global"] * 1.03
    stem = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)

    for mode, rottag in rottypes:
        is_local = (mode == "local")
        for module, modtag in modules:
            per_cap = allpanels[(mode, module)]
            cbar_w = 0.035 if is_local else 0.1
            extra_w = 0.0 if is_local else 0.8
            fig = plt.figure(
                figsize=(2 * NCOLS * 1.25 + 1.2 + extra_w, NROWS * 1.15 + 0.9),
                dpi=args.dpi)
            main_sf, cbar_sf = fig.subfigures(1, 2, width_ratios=[1.0, cbar_w], wspace=0)
            blk_sfs = main_sf.subfigures(1, 2, wspace=0)
            line_handles = line_labels = None
            for (caption, side), sf in zip(caps, blk_sfs):
                axes = sf.subplots(NROWS, NCOLS, sharex=True, sharey=True).ravel()
                for ax, (li, payload, xs) in zip(axes, per_cap[caption]):
                    ax.set_title(f"{li}", fontsize=10, pad=2)
                    if payload is None or not np.isfinite(payload).any():
                        ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                                transform=ax.transAxes, fontsize=8, color="0.5")
                        ax.set_xticks([]); ax.set_yticks([]); continue
                    if is_local:
                        r = payload.shape[0]
                        ax.imshow(payload, aspect="auto", origin="upper", cmap=CMAP,
                                  norm=norm_local, extent=[xs[0], xs[-1], r + 0.5, 0.5])
                        ax.set_yticks(range(1, r + 1))
                    else:
                        r = payload.shape[1]
                        for k in range(r):
                            c, ls = _kline(k)
                            ax.plot(xs, payload[:, k], color=c, linestyle=ls, lw=1.3,
                                    marker=".", ms=2.5, markeredgecolor="none",
                                    label=f"{k + 1}")
                        ax.set_ylim(0, ytop_global); ax.grid(True, alpha=0.3)
                        ax.set_axisbelow(True)
                        if line_handles is None:
                            line_handles, line_labels = ax.get_legend_handles_labels()
                    ax.tick_params(labelsize=8)
                grid = axes.reshape(NROWS, NCOLS)
                for row in range(NROWS):
                    for col in range(NCOLS):
                        a = grid[row, col]
                        if col != 0:
                            a.tick_params(labelleft=False)
                        if row != NROWS - 1:
                            a.tick_params(labelbottom=False)
                sf.subplots_adjust(top=0.94)
                sf.suptitle(caption, y=0.98, fontsize=22, fontweight="bold")
                if caption == "U":
                    sf.supylabel("principal angle index" if is_local
                                 else "principal angle (degrees)",
                                 x=-0.02, fontsize=13, fontweight="bold")
            main_sf.supxlabel("training step", y=0.02, fontsize=13, fontweight="bold")
            cax = cbar_sf.subplots(1, 1)
            if is_local:
                sm = ScalarMappable(norm=norm_local, cmap=CMAP); sm.set_array([])
                cbar = fig.colorbar(sm, cax=cax)
                cbar.set_label("degrees", fontsize=13); cbar.ax.tick_params(labelsize=8)
            else:
                cax.axis("off")
                if line_handles:
                    leg = cax.legend(line_handles, line_labels, title="k", fontsize=9,
                                     loc="center left", frameon=False, handlelength=1.4,
                                     borderaxespad=0.0)
                    leg.get_title().set_fontsize(11); leg.get_title().set_fontweight("bold")
            for ext in ("png", "pdf"):
                path = f"{stem}_{modtag}_{rottag}.{ext}"
                fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
                print(f"wrote {path}  ({module}, {mode}, {args.dpi} dpi, "
                      f"max {scale[mode]:.1f} deg)")
            plt.close(fig)


if __name__ == "__main__":
    main()