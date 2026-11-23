"""Local multi-GPU launcher for the experiment sweep.

Same selection flags, same resolvers, same config.build_payloads -> the SAME runs/ tree
written by run_task. Each payload runs in its own SUBPROCESS pinned to one GPU via
CUDA_VISIBLE_DEVICES, keeping N_GPUS workers busy until the queue drains. One run per
process = crash isolation: an OOM or CUDA fault kills that job's worker, not the sweep.

    python gpu_run.py --dry-run                                   # preview + full config
    python gpu_run.py --models qwen1.5b --tasks gsm8k --seeds 1 --kappas 10 --qs 10
    python gpu_run.py --models llama8b --tasks gsm8k,mtbench,humaneval \
                      --seeds 1,2,3 --kappas 0.01,1,10,100 --qs 0.01,1,10,100 --gpus 8

--------------------------------------------------------------------------------------
SPEC MODE (heterogeneous parallel sweep):
--------------------------------------------------------------------------------------
A single --adam-lr (or kappa/q) applies to a whole call, so a per-task-LR or per-cell
sweep would otherwise need many sequential calls that leave most GPUs idle. Instead pass
a JSON manifest -- a LIST of independent experiments, each its own dict of overrides:

    python gpu_run.py --spec experiments.json --models llama8b --seeds 1,2,3 \
                      --scale medium --gpus 8

    [
      {"tasks": "humaneval", "adam_lr": 2e-4, "kappas": "10",  "qs": "10"},
      {"tasks": "mmlu",      "adam_lr": 5e-5, "kappas": "1",   "qs": "1"},
      {"tasks": "gsm8k",     "adam_lr": 1e-4, "kappas": "100", "qs": "0.01", "coeff_optimizer": "adam"}
    ]

Any field a group omits falls back to the command-line default (so --models / --seeds /
--scale are set ONCE on the command line and each group only names what varies). All
payloads are merged into ONE queue and the dispatcher runs up to --gpus of them at once,
refilling a GPU the instant its job finishes. Per-group keys understood:
    tasks, seeds, scale, models, kappas, qs,
    adam_lr, coeff_optimizer, steps, preset, judge_model, no_periodic_eval, judge_later

Flags:
    --spec FILE     JSON manifest of heterogeneous experiments (this mode)
    --gpus N        how many GPUs to use (default: all visible, else 8)
    --out DIR       results tree root (default: ./runs)

Secrets come from the ENVIRONMENT:
    HF_TOKEN         for gated repos (Llama). Qwen is ungated.
    OPENAI_API_KEY   for the MT-Bench judge (and OPENAI_BASE_URL for a non-OpenAI endpoint).
Set them in your shell before launching (see SETUP.md).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # so `import grablora...` works when run from repo root


def _detect_gpus(requested: int) -> int:
    """How many GPUs to actually use. If --gpus given, honour it; else count via
    nvidia-smi; fall back to 8."""
    if requested > 0:
        return requested
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        n = len([ln for ln in out.splitlines() if ln.strip().startswith("GPU ")])
        return n or 8
    except Exception:
        return 8


def _csv(v) -> str:
    """Normalise a selection value to the comma-string the resolvers expect, so a spec
    may write either "2,3" or [2, 3]."""
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def _overrides_from(get) -> dict:
    """Build the RunConfig `overrides` dict from a getter `get(key, default)`. Used for
    BOTH the CLI namespace and a spec group."""
    ov = {}
    if get("preset", ""):
        ov["target_preset"] = get("preset", "")
    if get("steps", 0):
        ov["steps"] = int(get("steps", 0))
    if get("adam_lr", 0.0):
        ov["adam_lr"] = float(get("adam_lr", 0.0))
    if get("judge_model", ""):
        ov["judge_model"] = get("judge_model", "")
    if get("no_periodic_eval", False):
        ov["periodic_gen_eval"] = False
    if get("judge_later", False):
        ov["judge_later"] = True     # MT-Bench: generate now, judge offline later
    if get("coeff_optimizer", ""):
        ov["coeff_optimizer"] = get("coeff_optimizer", "")
    return ov


def _build_group(get, overwrite):
    """Expand ONE experiment (a getter over its fields) into (payloads, resolved).
    `get(key, default)` reads a spec-group key or, when absent, the CLI default."""
    from grablora.experiments.config import (resolve_tasks, resolve_methods,
                                             resolve_models, build_payloads,
                                             SHARED, SCALE)
    model_list = resolve_models(_csv(get("models", "qwen1.5b")))
    task_keys = resolve_tasks(_csv(get("tasks", "all")))
    method_list = resolve_methods(_csv(get("methods", "all")))
    seed_list = [int(s) for s in _csv(get("seeds", "1,2,3")).split(",") if s.strip()]
    kl = [float(x) for x in _csv(get("kappas", "10")).split(",") if x.strip()]
    ql = [float(x) for x in _csv(get("qs", "10")).split(",") if x.strip()]
    scale = get("scale", "small")
    overrides = _overrides_from(get)
    payloads = build_payloads(model_list, task_keys, method_list, seed_list, scale,
                              kappas=kl, qs=ql, overwrite=overwrite, overrides=overrides)
    resolved = {**SHARED, **SCALE[scale], **overrides,
                "models": model_list, "tasks": task_keys, "methods": method_list,
                "seeds": seed_list, "scale": scale, "kappas": kl, "qs": ql,
                "overwrite": overwrite}
    return payloads, resolved


def _resolve(args):
    """Resolve the run set. Returns (payloads, resolved_groups) where resolved_groups is
    a list of (label, resolved_dict, n_payloads). One entry in CLI mode; one per spec
    group in --spec mode. All payloads are merged into a single queue by the caller."""
    def _cli_get(k, d=None):
        return getattr(args, k, d)

    if not args.spec:
        payloads, resolved = _build_group(_cli_get, args.overwrite)
        return payloads, [("(cli)", resolved, len(payloads))]

    with open(args.spec) as f:
        groups = json.load(f)
    if not isinstance(groups, list):
        raise SystemExit(f"--spec {args.spec}: expected a JSON list of experiments")

    all_payloads, resolved_groups = [], []
    for i, group in enumerate(groups):
        # group key wins; otherwise fall back to the CLI default of the same name
        def _get(k, d=None, _g=group):
            return _g[k] if k in _g else getattr(args, k, d)
        pl, rs = _build_group(_get, args.overwrite)
        all_payloads.extend(pl)
        label = group.get("name") or f"exp{i}:{_csv(_get('tasks'))}/{_csv(_get('methods'))}"
        resolved_groups.append((label, rs, len(pl)))
    return all_payloads, resolved_groups


def _run_target(payload, base_out) -> str:
    """The run directory this payload writes to -- identity is method+task+seed (+bg
    kappa/q variant). Used for skip-if-exists AND duplicate detection."""
    from grablora.storage.run_writer import run_dir as _run_dir
    from grablora.tasks import get_task
    from grablora.experiments.run_task import RunConfig
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    cfg = RunConfig(**clean)
    task = get_task(cfg)
    ku, q0 = cfg.method_overrides.get("kappa_u"), cfg.method_overrides.get("q0")
    variant = f"k{ku:g}_q{q0:g}" if (ku is not None and q0 is not None) else ""
    return _run_dir(base_out, task.label, cfg.model_id, cfg.method, cfg.seed, variant)


def _is_done(payload, base_out) -> bool:
    """Skip-if-exists, matching run_task's own check: a summary.json at the run dir."""
    return os.path.exists(os.path.join(_run_target(payload, base_out), "summary.json"))


def _cell_name(p) -> str:
    variant = (f"/k{p.get('_kappa')}_q{p.get('_q')}" if p.get("_kind") == "bg" else "")
    return f"{p.get('_task', p.get('task'))}/{p['method']}/s{p['seed']}{variant}"


def _run_one(payload, base_out, gpu_id, logdir) -> dict:
    """Run ONE payload in a subprocess pinned to gpu_id (via worker.py, so a crash is
    contained). Returns a small status dict."""
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    job = {"payload": clean, "base_out": base_out}
    os.makedirs(logdir, exist_ok=True)
    tag = _cell_name(payload).replace("/", "__")
    job_path = os.path.join(logdir, f"{tag}.json")
    with open(job_path, "w") as f:
        json.dump(job, f)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # pin this run to one GPU
    log_path = os.path.join(logdir, f"{tag}.log")
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            [sys.executable, "-m", "grablora.experiments.worker", job_path],
            env=env, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    result_path = job_path + ".result"
    ok, err = False, None
    if os.path.exists(result_path):
        try:
            r = json.load(open(result_path))
            ok = bool(r.get("ok"))
            err = None if ok else str(r.get("error"))[:200]
        except Exception as e:
            err = f"unreadable result: {e}"
    else:
        err = f"no result file (exit {proc.returncode}); see {log_path}"
    return dict(cell=_cell_name(payload), gpu=gpu_id, ok=ok, seconds=round(dt, 1),
                error=err, log=log_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen1.5b")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--methods", default="all",
                    help="kept for compatibility; only GRAB-LoRA (bg_lora) is available")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--scale", default="small", choices=["small", "medium", "large"])
    ap.add_argument("--kappas", default="10")
    ap.add_argument("--qs", default="10")
    ap.add_argument("--preset", default="")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--adam-lr", dest="adam_lr", type=float, default=0.0)
    ap.add_argument("--coeff-optimizer", dest="coeff_optimizer", default="",
                    help="bg_lora coefficient estimator: 'kalman' (default) or 'adam'")
    ap.add_argument("--judge-model", dest="judge_model", default="")
    ap.add_argument("--no-periodic-eval", dest="no_periodic_eval", action="store_true")
    ap.add_argument("--judge-later", dest="judge_later", action="store_true",
                    help="MT-Bench: generate + save answers, skip judging; judge later with scripts/judge_mtbench.py")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    ap.add_argument("--spec", default="",
                    help="JSON manifest of heterogeneous experiments; each entry's fields "
                         "override the CLI defaults. Fans ALL of them across the GPUs at once.")
    ap.add_argument("--gpus", type=int, default=0, help="how many GPUs (default: all visible)")
    ap.add_argument("--out", default=os.environ.get("GRABLORA_OUT", "./runs"))
    args = ap.parse_args()

    base_out = os.path.abspath(args.out)
    n_gpus = _detect_gpus(args.gpus)
    payloads, resolved_groups = _resolve(args)

    # ---- print the resolved config(s) ----
    print("=" * 66)
    if args.spec:
        print(f"RESOLVED SPEC: {args.spec}  ({len(resolved_groups)} experiments)")
        for label, rs, n in resolved_groups:
            extra = [f"{k}={rs[k]}" for k in ("adam_lr", "coeff_optimizer") if k in rs]
            print(f"  [{n:3d}]  {label:42s} {'  '.join(extra)}")
    else:
        print("RESOLVED EXPERIMENT CONFIG (single source of truth -- config.py):")
        for k, v in resolved_groups[0][1].items():
            print(f"  {k:22s} {v}")
    print("-" * 66)
    print(f"TOTAL RUNS: {len(payloads)} GRAB-LoRA cells  "
          f"(skip-if-exists {'OFF' if args.overwrite else 'ON'})")
    print(f"GPUs: {n_gpus}   out: {base_out}")
    print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}   "
          f"OPENAI_API_KEY set: {bool(os.environ.get('OPENAI_API_KEY'))}")

    # ---- warn on colliding run targets (same dir; different LRs would clobber) ----
    seen, dups = {}, []
    for p in payloads:
        key = _run_target(p, base_out)
        if key in seen:
            dups.append((_cell_name(p), _cell_name(seen[key])))
        else:
            seen[key] = p
    if dups:
        print("-" * 66)
        print(f"WARNING: {len(dups)} payloads share a run dir with an earlier one. The run "
              f"dir does NOT encode the LR, so these would collide (last/skip wins):")
        for a, b in dups:
            print(f"   {a}   <->   {b}")
    print("=" * 66)

    if args.dry_run:
        print("dry run -- nothing launched.")
        for p in payloads[:40]:
            print("   ", _cell_name(p))
        if len(payloads) > 40:
            print(f"    ... and {len(payloads) - 40} more")
        return

    # ---- skip already-done cells (unless --overwrite) ----
    todo = payloads if args.overwrite else [p for p in payloads if not _is_done(p, base_out)]
    skipped = len(payloads) - len(todo)
    if skipped:
        print(f"skip-if-exists: {skipped} already done, {len(todo)} to run")
    if not todo:
        print("nothing to do -- all cells already have summary.json")
        return

    logdir = os.path.join(base_out, "_logs")
    # ---- dispatch: N_GPUS workers, one run per subprocess, each on a distinct GPU ----
    results = []
    done = 0
    t_start = time.time()
    print(f"\nlaunching {len(todo)} runs across {n_gpus} GPUs "
          f"(up to {n_gpus} different configs at once)...\n")

    free_gpus = list(range(n_gpus))
    with ThreadPoolExecutor(max_workers=n_gpus) as ex:
        futures = {}
        queue = list(todo)
        # prime the pool: one job per free GPU
        while queue and free_gpus:
            p = queue.pop(0)
            g = free_gpus.pop(0)
            futures[ex.submit(_run_one, p, base_out, g, logdir)] = g
        # as each finishes, free its GPU and submit the next queued job on it
        while futures:
            for fut in as_completed(list(futures)):
                g = futures.pop(fut)
                r = fut.result()
                results.append(r)
                done += 1
                status = "ok " if r["ok"] else "FAIL"
                print(f"[{done}/{len(todo)}] {status} {r['cell']:45s} "
                      f"gpu{r['gpu']} {r['seconds']:6.0f}s"
                      + (f"  ERR {r['error']}" if not r["ok"] else ""))
                if queue:
                    p = queue.pop(0)
                    futures[ex.submit(_run_one, p, base_out, g, logdir)] = g
                break   # re-enter as_completed with the updated futures set

    # ---- summary ----
    n_ok = sum(1 for r in results if r["ok"])
    n_fail = len(results) - n_ok
    print("\n" + "=" * 66)
    print(f"DONE: {n_ok} ok, {n_fail} failed, {skipped} skipped  "
          f"in {(time.time()-t_start)/60:.1f} min")
    if n_fail:
        print("\nfailures (see logs):")
        for r in results:
            if not r["ok"]:
                print(f"  {r['cell']}  --  {r['error']}")
    print(f"\nresults tree: {base_out}")
    print("summarize:  python scripts/summarize.py --runs", base_out)


if __name__ == "__main__":
    main()