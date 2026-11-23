#!/usr/bin/env python
"""Paper-style summary table: mean +/- std over seeds, across methods x tasks x models.

Reads the same runs/ tree as gpu_run.py / inventory.py and pivots the results into a
table you can read straight into a paper: one row per method (bg_lora cells expanded
per kappa/q), one column per task, each cell = mean +/- std of the task metric over
seeds. Groups models separately (so T5/GLUE and Qwen/GSM8K don't mix), which is what
you want when different tasks run on different models.

    python scripts/analyze.py                                # current runs on disk
    python scripts/analyze.py --models qwen7b --tasks mmlu   # mmlu -> mmlu_5shot only
    python scripts/analyze.py --models qwen7b,llama8b --tasks gsm8k,mmlu,humaneval
    python scripts/analyze.py --metric final_test_metric --csv table.csv
    python scripts/analyze.py --tasks glue --pct             # scale metric x100 (accuracy %)

TASK SELECTION follows the archive convention: a task folder ending in `_v<N>`
(e.g. mmlu_5shot_v1, gsm8k_v2) is treated as an OLD/archived run and hidden by
default, so the table shows only current results. Matching is convention-aware:

  * `--tasks gsm8k`      -> exactly `gsm8k`            (exact name wins; no `_v*`)
  * `--tasks mmlu`       -> `mmlu_5shot`               (base/shorthand -> current;
                                                        excludes `mmlu_5shot_v*`)
  * `--tasks mmlu_5shot_v1` -> that exact archived run (name a version explicitly
                                                        to include it)
  * `--with-versions`    -> stop hiding `_v<N>` runs   (old merge-everything table)

Models/methods still use the project resolvers, so short names / groups work
(`--models qwen7b` -> Qwen/Qwen2.5-7B).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "scripts" else HERE
sys.path.insert(0, ROOT)

# a task folder whose name ends in _v1, _v2, ... is an archived/older pivot run
_VERSION_RE = re.compile(r"_v\d+$")


def _slug(s):
    return str(s).replace("/", "--").replace(" ", "_")


def _is_versioned(name):
    return bool(_VERSION_RE.search(str(name)))


def _task_dir_name(s):
    """Real task identity for a summary, taken from its FOLDER name rather than the
    `task` field. The field is written at run time and does NOT change when a folder
    is later renamed for archiving (e.g. mmlu_5shot -> mmlu_5shot_v1), so all versions
    still report task="mmlu_5shot" internally and can't be told apart by the field.

    The distinguishing name is the path segment that equals the task, or the task
    plus a `_v<N>` suffix. Returns that segment (deepest match), so mmlu_5shot,
    mmlu_5shot_v1 and mmlu_5shot_v2 become three distinct task identities again.
    Falls back to the `task` field if no path segment matches."""
    task = str(s.get("task", ""))
    segs = str(s.get("_dir", "")).split(os.sep)
    for seg in reversed(segs):                       # deepest match wins
        if seg == task or _VERSION_RE.sub("", seg) == task:
            return seg
    return task


def _tokens(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _resolvers():
    try:
        from grablora.experiments.config import (resolve_models, resolve_tasks,
                                                 resolve_methods)
        return resolve_models, resolve_tasks, resolve_methods
    except Exception:
        ident = lambda s: [x.strip() for x in (s or "").split(",") if x.strip()]
        return ident, ident, (lambda s, b=False: ident(s))


def _select(typed, resolver, present, slugify=False):
    """Model/method selection, EXACT-FIRST: a token that exactly names a value on
    disk is matched literally; anything else goes through the project resolver so
    short names / group aliases still expand (qwen7b -> Qwen/Qwen2.5-7B). Returns
    None when `typed` is empty (no filter on this dimension)."""
    if not typed:
        return None
    want = set()
    for tok in _tokens(typed):
        key = _slug(tok) if slugify else tok
        if key in present:
            want.add(key)
        else:
            for r in resolver(tok):
                want.add(_slug(r) if slugify else r)
    return want


def _select_tasks(typed, present, with_versions):
    """Convention-aware task selection over the task names present on disk.

    For each token:
      1. if it exactly names a task on disk, take that (and only that) -- this lets
         you pin an exact name, versioned or not (e.g. `mmlu_5shot`, `mmlu_5shot_v1`);
      2. otherwise treat it as a base/shorthand and match any current task equal to
         it or starting with `<token>_` (so `mmlu` -> `mmlu_5shot`, `glue` -> glue_*),
         skipping archived `_v<N>` runs unless --with-versions is set.

    Returns None when `typed` is empty.
    """
    if not typed:
        return None
    want = set()
    for tok in _tokens(typed):
        exact = {d for d in present if d == tok}
        if exact:
            want |= exact
            continue
        for d in present:
            if _is_versioned(d) and not with_versions:
                continue
            if d == tok or d.startswith(tok + "_"):
                want.add(d)
    return want


def _load(base_out):
    out = []
    for p in glob.glob(os.path.join(base_out, "**", "summary.json"), recursive=True):
        try:
            s = json.load(open(p))
            s["_dir"] = os.path.dirname(p)
            out.append(s)
        except Exception:
            pass
    return out


def _mean_std(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and x == x]
    if not v:
        return float("nan"), float("nan"), 0
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v)) if len(v) > 1 else 0.0
    return m, sd, len(v)


def _method_label(s):
    """Row label for a summary. bg_lora cells expand to bg_lora(k=..,q=..) so each
    grid point is its own row; baselines use the bare method name."""
    method = str(s.get("method", ""))
    k, q = s.get("kappa"), s.get("q")
    if method == "bg_lora" and k is not None and q is not None:
        return f"bg_lora(k={k:g},q={q:g})"
    return method


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("GRABLORA_OUT", "./runs"))
    ap.add_argument("--models", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--methods", default="")
    ap.add_argument("--metric", default="final_test_metric")
    ap.add_argument("--pct", action="store_true", help="multiply metric by 100 (report as %)")
    ap.add_argument("--csv", default="", help="write the table as CSV")
    ap.add_argument("--min-seeds", type=int, default=1,
                    help="only show cells with at least this many seeds")
    ap.add_argument("--with-versions", action="store_true",
                    help="include archived `_v<N>` task runs (default: hidden). "
                         "Turns the table back into the old merge-everything view.")
    args = ap.parse_args()

    base_out = os.path.abspath(args.out)
    resolve_models, resolve_tasks, resolve_methods = _resolvers()

    summaries = _load(base_out)
    if not summaries:
        print(f"no summary.json under {base_out}")
        return

    present_models = {_slug(str(s.get("model", ""))) for s in summaries}
    present_tasks = {_task_dir_name(s) for s in summaries}   # folder-derived identity
    present_methods = {str(s.get("method", "")) for s in summaries}

    want_models = _select(args.models, resolve_models, present_models, slugify=True)
    want_methods = _select(args.methods, lambda s: resolve_methods(s, False),
                           present_methods)
    want_tasks = _select_tasks(args.tasks, present_tasks, args.with_versions)

    def _task_ok(task):
        if want_tasks is not None:
            return task in want_tasks
        # no --tasks filter: show current runs, hide archived _v<N> unless asked
        return args.with_versions or not _is_versioned(task)

    # collect metric values keyed by (model, method_label, task) -> [per-seed values]
    cells = defaultdict(list)
    metric_names = {}                       # task -> its metric_name (for the header)
    scale = 100.0 if args.pct else 1.0
    for s in summaries:
        task = _task_dir_name(s)            # folder-derived: distinguishes _v<N>
        model = str(s.get("model", ""))
        if not _task_ok(task):
            continue
        if want_models is not None and _slug(model) not in want_models:
            continue
        if want_methods is not None and str(s.get("method", "")) not in want_methods:
            continue
        val = s.get(args.metric)
        if not isinstance(val, (int, float)) or val != val:
            continue                        # skip missing/NaN
        label = _method_label(s)
        cells[(model, label, task)].append(val * scale)
        metric_names.setdefault(task, s.get("metric_name", ""))

    if not cells:
        print("no runs matched the filters.")
        return

    # organise per model: rows = method labels, cols = tasks
    models = sorted({k[0] for k in cells})
    for model in models:
        tasks = sorted({k[2] for k in cells if k[0] == model})
        methods = sorted({k[1] for k in cells if k[0] == model})

        print("\n" + "=" * 78)
        print(f"MODEL: {model}     metric: {args.metric}"
              + ("  (x100)" if args.pct else ""))
        print("tasks: " + ", ".join(f"{t}[{metric_names.get(t,'')}]" for t in tasks))
        print("=" * 78)

        mw = max([len("method")] + [len(m) for m in methods])
        colw = 16
        header = f"{'method':<{mw}}  " + "  ".join(f"{t:>{colw}}" for t in tasks)
        print(header)
        print("-" * len(header))

        for method in methods:
            parts = [f"{method:<{mw}}"]
            for t in tasks:
                vals = cells.get((model, method, t), [])
                m, sd, n = _mean_std(vals)
                if n >= args.min_seeds and n > 0:
                    fmt = f"{m:.3f}±{sd:.3f}(n{n})" if not args.pct else f"{m:.2f}±{sd:.2f}(n{n})"
                else:
                    fmt = "--"
                parts.append(f"{fmt:>{colw}}")
            print("  ".join(parts))
        print("-" * len(header))

    # ---- CSV (long form: one row per model,method,task) ----
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "method", "task", "metric_name", "mean", "std",
                        "n_seeds", "per_seed"])
            for (model, method, task), vals in sorted(cells.items()):
                m, sd, n = _mean_std(vals)
                w.writerow([model, method, task, metric_names.get(task, ""),
                            f"{m:.6f}", f"{sd:.6f}", n,
                            ";".join(f"{v:.6f}" for v in vals)])
        print(f"\nCSV -> {args.csv}")


if __name__ == "__main__":
    main()