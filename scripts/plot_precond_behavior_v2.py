#!/usr/bin/env python3
"""Draw how the edited-Adam preconditioner behaves, from bg_lora_adam precond_trace.jsonl.

Two ways to pick the runs to overlay:
  (1) CLI (recommended for a q-sweep at fixed LR+k):
        python plot_precond_behavior.py \
          --runs runs/glue_sst2/google-t5--t5-base/bg_lora_adam/seed_6__k0.1_q0.01/precond_trace.jsonl \
                 runs/glue_sst2/google-t5--t5-base/bg_lora_adam/seed_6__k0.1_q0.001/precond_trace.jsonl \
                 ... (5 q values) \
          --labels "q=1e-2" "q=1e-3" "q=1e-4" "q=1e-5" "q=1e-6" \
          --output-dir figures/preconditioner
  (2) edit the CONFIG block below and run with no args.

Each precond_trace.jsonl row is one atom at one refresh:
  m/abs_m  : the coefficient STEP (NB: clamped by the step cap -> may sit flat at the cap)
  gain     : 1/(omega+eps)          OUR preconditioner multiplier   (uncapped)
  gain_adam: 1/(omega_adam+eps)     plain-Adam multiplier at the same atom
  mu_hat   : Adam first moment (the numerator; step = mu_hat * gain, then capped)
  omega=sqrt(obs_var+P_pred);  obs_var=R (evidence noise 1/(n*beta));  P_pred=P^- (=rho^2 P + q)
  q        : process noise q0+q1*(1-rho^2);   rho, theta_V : rotation

Four panels (values aggregated as the MEDIAN over atoms+layers per step, one line per config):
  1. step size: |m| (capped) and mu_hat (raw 1st moment) over training  -> shows the cap
  2. our gain vs plain-Adam gain over training  (ours solid, adam dashed) -> the preconditioner effect
  3. omega^2 parts: obs_var (R) and P_pred (P^-) and q over training (log) -> what sets the denominator
  4. gain vs theta_V (refresh rows) -> rotation-aware damping (uncertainty-aware step)
"""
from __future__ import annotations

# ============================ CONFIG (used only when --runs is NOT given) ======
RUNS_ROOT = "runs"
TASK      = "gsm8k"          # task label dir under runs/ (e.g. "gsm8k", "math", "glue_sst2")
REGIME    = "bg_lora_adam"   # method+regime dir (adam Option C)
# one entry per run to overlay; label is what shows in the legend
CONFIGS = [
    dict(seed=1, k=0.1, q=1e-4, label="s1  k0.1 q1e-4"),
    dict(seed=2, k=0.1, q=1e-6, label="s2  k0.1 q1e-6"),
]
PHASE_FOR_STEP = "all"       # "all" | "refresh" | "init"  (panels 1-3)
LAYER_CONTAINS = ""          # e.g. "q_proj" to restrict to one projection; "" = all layers
OUT = "precond_behavior.png"
# =========================================================================

import argparse
import glob
import json
import os


def _variant(k, q):
    return f"k{k:g}_q{q:g}"          # matches gpu_run's run-dir naming: seed_<s>__k<k>_q<q>


def _find(cfg):
    pat = os.path.join(RUNS_ROOT, TASK, "*", REGIME,
                       f"seed_{cfg['seed']}__{_variant(cfg['k'], cfg['q'])}", "precond_trace.jsonl")
    hits = sorted(glob.glob(pat))
    if not hits:
        # looser fallback: match seed + k, any q formatting
        pat2 = os.path.join(RUNS_ROOT, TASK, "*", REGIME,
                            f"seed_{cfg['seed']}__k{cfg['k']:g}_q*", "precond_trace.jsonl")
        hits = sorted(glob.glob(pat2))
    return hits[0] if hits else None


def _rows(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _series_from_cli(runs, labels, layer_contains):
    """Build (label, rows, path) triples directly from explicit --runs paths (glob ok)."""
    series = []
    for i, spec in enumerate(runs):
        hits = sorted(glob.glob(spec)) if any(c in spec for c in "*?[") else [spec]
        hits = [h for h in hits if os.path.exists(h)]
        if not hits:
            print(f"[warn] no file for {spec!r} -- skipping")
            continue
        path = hits[0]
        if len(hits) > 1:
            print(f"[info] {spec!r} matched {len(hits)}; using {path}")
        label = labels[i] if labels and i < len(labels) else \
            os.path.basename(os.path.dirname(path))       # e.g. seed_6__k0.1_q0.01
        rows = _rows(path)
        if layer_contains:
            rows = [r for r in rows if layer_contains in str(r.get("layer", ""))]
        series.append((label, rows, path))
    return series


def _series_from_config(layer_contains):
    series = []
    for cfg in CONFIGS:
        p = _find(cfg)
        if not p:
            print(f"[warn] no precond_trace for {cfg['label']} "
                  f"(seed {cfg['seed']}, {_variant(cfg['k'], cfg['q'])}) -- skipping")
            continue
        rows = _rows(p)
        if layer_contains:
            rows = [r for r in rows if layer_contains in str(r.get("layer", ""))]
        series.append((cfg["label"], rows, p))
    return series


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None,
                    help="explicit precond_trace.jsonl paths (glob ok); overrides the CONFIG block")
    ap.add_argument("--labels", nargs="*", default=None, help="legend label per run")
    ap.add_argument("--output-dir", default=None, help="directory to write precond_behavior.{png,pdf}")
    ap.add_argument("--out", default=None, help="explicit output PNG path (overrides --output-dir)")
    ap.add_argument("--title", default=None, help="figure suptitle (default derived)")
    ap.add_argument("--phase-for-step", dest="phase_for_step", default=PHASE_FOR_STEP,
                    choices=["all", "refresh", "init"])
    ap.add_argument("--layer-contains", dest="layer_contains", default=LAYER_CONTAINS,
                    help='restrict to layers whose key contains this (e.g. "q_proj"); "" = all')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    phase_for_step = args.phase_for_step
    if args.runs:
        series = _series_from_cli(args.runs, args.labels, args.layer_contains)
        title = args.title or "Edited-Adam preconditioner behavior"
    else:
        series = _series_from_config(args.layer_contains)
        title = args.title or f"Edited-Adam preconditioner behavior — {TASK}"
    for label, rows, p in series:
        capped = np.mean([abs(r.get("m", 0)) >= 0.999 for r in rows]) if rows else float("nan")
        print(f"  {label:22s} rows={len(rows):6d}  |m|>=~cap frac={capped:.2%}  <- {p}")
    if not series:
        raise SystemExit("no usable precond traces (check the --runs paths or the CONFIG block)")

    # resolve output path
    if args.out:
        out = args.out
    elif args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, "precond_behavior.png")
    else:
        out = OUT

    def agg(rows, key, phase):
        """median of `key` per step over atoms/layers, for the given phase."""
        by = {}
        for r in rows:
            if phase != "all" and r.get("phase") != phase:
                continue
            v = r.get(key)
            if isinstance(v, (int, float)) and v == v:
                by.setdefault(int(r["step"]), []).append(float(v))
        steps = sorted(by)
        return np.array(steps, float), np.array([np.median(by[s]) for s in steps], float)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a1, a2, a3, a4 = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    for i, (label, rows, _) in enumerate(series):
        c = cmap(i % 10)
        # 1. step size: |m| (capped) solid, mu_hat (raw 1st moment) dashed
        s, y = agg(rows, "abs_m", phase_for_step)
        if len(s): a1.plot(s, y, "-", color=c, lw=1.8, label=f"{label}  |m|")
        s, y = agg(rows, "mu_hat", phase_for_step)
        if len(s): a1.plot(s, np.abs(y), "--", color=c, lw=1.0, alpha=0.7, label=f"{label}  |mu_hat|")
        # 2. our gain vs adam gain
        s, y = agg(rows, "gain", phase_for_step)
        if len(s): a2.plot(s, y, "-", color=c, lw=1.8, label=f"{label}  ours")
        s, y = agg(rows, "gain_adam", phase_for_step)
        if len(s): a2.plot(s, y, "--", color=c, lw=1.0, alpha=0.7, label=f"{label}  adam")
        # 3. denominator parts
        for key, ls, lab in (("obs_var", "-", "R=obs_var"), ("P_pred", "--", "P_pred"), ("q", ":", "q")):
            s, y = agg(rows, key, phase_for_step)
            if len(s): a3.plot(s, np.clip(y, 1e-12, None), ls, color=c, lw=1.6 if ls == "-" else 1.1,
                               alpha=0.9, label=f"{label} {lab}")
        # 4. gain vs theta_V (refresh only, rotation-aware damping)
        th, g = [], []
        for r in rows:
            if r.get("phase") == "refresh" and r.get("theta_V") is not None and r.get("gain") is not None:
                tv = r["theta_V"]
                if tv == tv:
                    th.append(tv); g.append(r["gain"])
        if th:
            a4.scatter(th, np.clip(g, 1e-12, None), s=8, alpha=0.35, color=c, edgecolors="none", label=label)

    a1.set_title("step size over training  (|m| capped, |mu_hat| raw)")
    a1.set_xlabel("step"); a1.set_ylabel("magnitude"); a1.set_yscale("log")
    a1.grid(True, alpha=0.25); a1.legend(fontsize=7)
    a2.set_title("preconditioner gain: ours vs plain Adam")
    a2.set_xlabel("step"); a2.set_ylabel("gain = 1/(sqrt(var)+eps)"); a2.set_yscale("log")
    a2.grid(True, alpha=0.25); a2.legend(fontsize=7)
    a3.set_title("denominator parts:  omega^2 = R + P_pred  (P_pred = rho^2 P + q)")
    a3.set_xlabel("step"); a3.set_ylabel("variance"); a3.set_yscale("log")
    a3.grid(True, alpha=0.25); a3.legend(fontsize=7)
    a4.set_title("rotation-aware damping: gain vs theta_V (refresh)")
    a4.set_xlabel("theta_V (rad)"); a4.set_ylabel("gain"); a4.set_yscale("log")
    a4.grid(True, alpha=0.25); a4.legend(fontsize=7)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    if out.endswith(".png"):
        pdf = out[:-4] + ".pdf"
        fig.savefig(pdf, bbox_inches="tight")
        print("wrote", pdf)


if __name__ == "__main__":
    raise SystemExit(main())