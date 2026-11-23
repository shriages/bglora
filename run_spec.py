#!/usr/bin/env python3
r"""Run a JSON experiment manifest (a list of fully-resolved run configs) through the same
tested worker dispatch gpu_run.py uses -- one run per subprocess, pinned round-robin to GPUs.

The manifest is either:
  * {"meta": {...}, "runs": [ <payload>, ... ]}   (what the generator writes), or
  * a bare list [ <payload>, ... ].
Each <payload> is a RunConfig-override dict (build_payloads output); keys starting with "_"
are ignored at dispatch. `overwrite` is read from each payload (skip-if-exists honours it),
and --overwrite / --no-overwrite on the CLI force it for the whole manifest.

Usage
-----
  python run_spec.py experiments.json --dry-run
  python run_spec.py experiments.json --gpus 8
  python run_spec.py experiments.json --gpus 8 --no-overwrite     # skip finished cells
  python run_spec.py experiments.json --gpus 8 --precond          # record precond traces
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gpu_run                                          # tested worker dispatch / GPU pinning


def _load(path):
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return obj.get("runs", []), obj.get("meta", {})
    if isinstance(obj, list):
        return obj, {}
    raise SystemExit(f"{path}: expected a list or a {{'runs': [...]}} object")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="path to the JSON manifest (e.g. experiments.json)")
    ap.add_argument("--gpus", type=int, default=0, help="how many GPUs (default: all visible)")
    ap.add_argument("--out", default=os.environ.get("GRABLORA_OUT", "./runs"))
    ap.add_argument("--overwrite", dest="overwrite", action="store_true", default=None,
                    help="force overwrite for every cell (else use each payload's own flag)")
    ap.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                    help="force skip-if-exists for every cell")
    ap.add_argument("--precond", action="store_true",
                    help="record precond_trace.jsonl (sets GRABLORA_PRECOND_TRACE=1)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    args = ap.parse_args()

    if args.precond:
        os.environ["GRABLORA_PRECOND_TRACE"] = "1"      # inherited by every worker subprocess

    runs, meta = _load(args.manifest)
    if not runs:
        raise SystemExit(f"{args.manifest}: no runs")
    # CLI overwrite (if given) wins over each payload's own flag
    if args.overwrite is not None:
        for p in runs:
            p["overwrite"] = args.overwrite
    overwrite_all = all(p.get("overwrite") for p in runs)

    base_out = os.path.abspath(args.out)
    n_gpus = gpu_run._detect_gpus(args.gpus)

    print("=" * 72)
    print(f"run_spec -- {os.path.basename(args.manifest)}"
          + (f"   ({meta.get('name','')} {meta.get('task','')} "
             f"scale={meta.get('scale','')} {meta.get('regime','')})" if meta else ""))
    print("-" * 72)
    print(f"TOTAL RUNS: {len(runs)}   GPUs: {n_gpus}   out: {base_out}   "
          f"overwrite: {'ON' if overwrite_all else 'mixed/OFF'}   "
          f"precond: {'ON' if args.precond else 'OFF'}")
    print("=" * 72)
    if args.dry_run:
        for p in runs[:24]:
            print("   ", gpu_run._cell_name(p))
        if len(runs) > 24:
            print(f"    ... and {len(runs) - 24} more")
        return

    todo = [p for p in runs if p.get("overwrite") or not gpu_run._is_done(p, base_out)]
    skipped = len(runs) - len(todo)
    if skipped:
        print(f"skip-if-exists: {skipped} already done, {len(todo)} to run")
    if not todo:
        print("nothing to do -- all cells already have summary.json")
        return

    logdir = os.path.join(base_out, "_logs")
    results, done, t0 = [], 0, time.time()
    print(f"\nlaunching {len(todo)} runs across {n_gpus} GPUs"
          f"{' (precond ON)' if args.precond else ''}...\n")
    free_gpus = list(range(n_gpus))
    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        futures, queue = {}, list(todo)
        while queue and free_gpus:
            futures[ex.submit(gpu_run._run_one, queue.pop(0), base_out,
                              free_gpus.pop(0), logdir)] = True
        while futures:
            for fut in as_completed(list(futures)):
                del futures[fut]
                r = fut.result()
                results.append(r)
                done += 1
                status = "ok " if r["ok"] else "FAIL"
                print(f"[{done}/{len(todo)}] {status} {r['cell']:46s} gpu{r['gpu']} "
                      f"{r['seconds']:6.0f}s" + (f"  ERR {r['error']}" if not r["ok"] else ""))
                if queue:
                    futures[ex.submit(gpu_run._run_one, queue.pop(0), base_out,
                                      r["gpu"], logdir)] = True
                break

    n_ok = sum(1 for r in results if r["ok"])
    print("\n" + "=" * 72)
    print(f"DONE: {n_ok} ok, {len(results) - n_ok} failed, {skipped} skipped  "
          f"in {(time.time() - t0) / 60:.1f} min")
    for r in results:
        if not r["ok"]:
            print(f"  FAIL {r['cell']}  --  {r['error']}")
    print(f"\nresults tree: {base_out}")
    print("summarize:  python scripts/summarize.py --runs", base_out)


if __name__ == "__main__":
    main()