from __future__ import annotations

import argparse
import csv
import json
import os


# ---- defaults: same tree pattern as gsm8k, just a different benchmark folder ----
RUNS_ROOT = os.environ.get("GRABLORA_OUT", "./runs")
MODEL_TAG = "meta-llama--Llama-3.1-8B"
# benchmark folder -> metric label reported for it
BENCHMARKS = {
    "humaneval": "pass@1",
    "mmlu_5shot": "accuracy",
}


def _read_jsonl(path):
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


def _parse_leaf(leaf):
    """seed_<seed>[__<variant>] -> (seed_str, variant)."""
    body = leaf[len("seed_"):] if leaf.startswith("seed_") else leaf
    if "__" in body:
        seed, variant = body.split("__", 1)
    else:
        seed, variant = body, ""
    return seed, variant


def _score(run_dir, tag):
    """Fraction-correct for one run: recompute from predictions, else summary.json.
    Returns (score, source) with score in [0,1] or (None, 'missing')."""
    pred = os.path.join(run_dir, "predictions", f"{tag}.jsonl")
    if os.path.exists(pred):
        rows = _read_jsonl(pred)
        # per-row boolean correctness flag (humaneval: 'passed'; others: 'correct')
        for flag in ("correct", "passed"):
            if rows and all(flag in r for r in rows):
                return sum(bool(r[flag]) for r in rows) / len(rows), "predictions"
        # or an explicit pred/gold pair
        if rows and all(("pred" in r and "gold" in r) for r in rows):
            return sum(str(r["pred"]) == str(r["gold"]) for r in rows) / len(rows), "predictions"
    summ = os.path.join(run_dir, "summary.json")
    if os.path.exists(summ):
        try:
            s = json.load(open(summ))
            v = s.get("final_test_metric")
            if v is not None:
                return float(v), "summary.json"
        except Exception:
            pass
    return None, "missing"


def process_benchmark(model_dir, metric, seeds, tag, out, sort):
    """Collect every method under model_dir, average over seeds, write one CSV. Returns records."""
    import numpy as np

    if not os.path.isdir(model_dir):
        print(f"[warn] model dir not found, skipping: {model_dir}")
        return []
    seeds = [str(s) for s in seeds]

    methods = sorted(d for d in os.listdir(model_dir)
                     if os.path.isdir(os.path.join(model_dir, d)))
    if not methods:
        print(f"[warn] no method subdirs under {model_dir}")
        return []

    records = []
    for method in methods:
        mdir = os.path.join(model_dir, method)
        is_bg = method.startswith("bg_lora")
        per_seed = {}          # seed -> (score, source, run_dir)
        for leaf in sorted(os.listdir(mdir)):
            ld = os.path.join(mdir, leaf)
            if not os.path.isdir(ld) or not leaf.startswith("seed_"):
                continue
            seed, variant = _parse_leaf(leaf)
            if seed not in seeds:
                continue
            # bg_lora has a single configuration here -> no k/q filter; take whatever leaf exists.
            sc, src = _score(ld, tag)
            if sc is None:
                print(f"[warn] no metric for {method}/{leaf} -- skipping")
                continue
            if seed in per_seed:
                print(f"[warn] {method} seed {seed}: multiple leaves match "
                      f"({per_seed[seed][2]} and {ld}); keeping first")
                continue
            per_seed[seed] = (sc, src, ld)
        if not per_seed:
            print(f"[warn] {method}: no runs for seeds {seeds} -- skipping")
            continue

        found = [s for s in seeds if s in per_seed]
        vals = np.array([per_seed[s][0] for s in found], float)
        # one bg variant only; report it if present so the CSV records which config was used
        bg_variant = ""
        if is_bg:
            variants = {_parse_leaf(os.path.basename(per_seed[s][2]))[1] for s in found}
            bg_variant = ";".join(sorted(v for v in variants if v))
        rec = {
            "method": method,
            "regime": "adam" if method.endswith("_adam") else "budget",
            "variant": bg_variant,
            "n_seeds": len(found),
            "seeds_found": ",".join(found),
            "metric": metric,
            "score_mean": float(vals.mean()),
            "score_std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "score_min": float(vals.min()),
            "score_max": float(vals.max()),
            "score_pct": float(vals.mean() * 100.0),
            "source": ";".join(sorted({per_seed[s][1] for s in found})),
        }
        for s in seeds:
            rec[f"score_seed{s}"] = (float(per_seed[s][0]) if s in per_seed else "")
        records.append(rec)

    if not records:
        print(f"[warn] {model_dir}: no methods produced usable seed runs")
        return []

    if sort == "score":
        records.sort(key=lambda r: r["score_mean"], reverse=True)
    else:
        records.sort(key=lambda r: r["method"])

    cols = (["method", "regime", "variant", "n_seeds", "seeds_found", "metric",
             "score_mean", "score_std", "score_min", "score_max", "score_pct"]
            + [f"score_seed{s}" for s in seeds] + ["source"])
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow([f"{r[c]:.4f}" if isinstance(r.get(c), float) else r.get(c, "")
                        for c in cols])

    # readable console table
    seed_hdr = "  ".join(f"s{s}" for s in seeds)
    print(f"\n{os.path.basename(model_dir)}  |  {metric}  averaged over seeds {seeds}\n")
    print(f"{'method':26s} {'n':>2}  {'mean':>7} {'std':>7}   {seed_hdr}")
    print("-" * (50 + 6 * len(seeds)))
    for r in records:
        cells = "  ".join(
            (f"{r[f'score_seed{s}']:.3f}" if isinstance(r.get(f'score_seed{s}'), float) else "  -  ")
            for s in seeds)
        print(f"{r['method']:26s} {r['n_seeds']:>2}  {r['score_mean']:.4f} "
              f"{r['score_std']:.4f}   {cells}")
    print(f"\nwrote {out}")
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=None,
                    help="a single benchmark model dir (overrides the default two)")
    ap.add_argument("--metric", default=None,
                    help="metric label for --model-dir (e.g. pass@1, accuracy)")
    ap.add_argument("--runs-root", default=RUNS_ROOT, help="root holding the benchmark folders")
    ap.add_argument("--model-tag", default=MODEL_TAG, help="model subdir name under each benchmark")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="seeds to average over (default 1 2 3)")
    ap.add_argument("--tag", default="final", help="predictions file tag (default: final)")
    ap.add_argument("--out-dir", default="figures", help="dir for the per-benchmark CSVs")
    ap.add_argument("--out", default=None, help="explicit CSV path (only with --model-dir)")
    ap.add_argument("--sort", default="score", choices=["score", "method"],
                    help="row order: by mean score desc (default) or method name")
    args = ap.parse_args(argv)

    if args.model_dir:
        metric = args.metric or "score"
        out = args.out or os.path.join(args.out_dir, "bench_average.csv")
        process_benchmark(args.model_dir, metric, args.seeds, args.tag, out, args.sort)
        return 0

    for bench, metric in BENCHMARKS.items():
        model_dir = os.path.join(args.runs_root, bench, args.model_tag)
        out = os.path.join(args.out_dir, f"{bench}_average.csv")
        process_benchmark(model_dir, metric, args.seeds, args.tag, out, args.sort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())