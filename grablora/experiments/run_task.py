"""Run ONE (task, model, method, seed) and write the structured run directory
(storage/run_writer.RunWriter):

    runs/{task}/{model}/{method}/seed_{seed}/
        config.yaml environment.json dataset_manifest.json
        train_metrics.jsonl eval_metrics.jsonl resource_metrics.jsonl
        geometry/checkpoint_{step}.parquet  predictions/  summary.json

Trained with decoupled AdamW at a real learning rate (annealed schedule) and early
stopping on a val metric; the task metric + val_ce + per-layer geometry are recorded at
every checkpoint in `eval_steps`.
"""
from __future__ import annotations

import os
# Reduce CUDA fragmentation OOM near the memory ceiling (e.g. 40GB A100). Must be set
# before the caching allocator initialises (before the first CUDA alloc), so it lives at
# the top of this module, ahead of any model load. setdefault -> a shell that already set
# it still wins. Harmless on larger cards.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
from dataclasses import dataclass, field

import torch

from ..core import MethodSpec
from ..methods import build, METHODS
from ..models import load_model_and_tokenizer, find_targets, freeze_and_arm
from ..models.loading import is_seq2seq
from ..train.lm_loop import train_lm
from ..train.schedule import make_schedule
from ..eval.gsm8k_eval import val_ce, _free_cuda
from ..tasks import get_task, tokenize_pairs, tokenize_pairs_seq2seq
from ..storage.run_writer import RunWriter, content_hash
from . import geometry as geo


def _flush_precond(run_dir_path, rows):
    """Append per-atom preconditioner-trace rows (from bg_lora.pop_precond_trace) to
    <run_dir>/precond_trace.jsonl. No-op when tracing is off (rows empty)."""
    if not rows:
        return
    import json as _json
    with open(os.path.join(run_dir_path, "precond_trace.jsonl"), "a") as f:
        for r in rows:
            f.write(_json.dumps(r) + "\n")


@dataclass
class RunConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B"
    revision: str = None
    task: str = "gsm8k"                  # gsm8k | math | glue | mmlu | humaneval | mtbench
    glue_subset: str = "sst2"            # used when task == "glue"
    # per-task eval knobs (ignored by tasks that don't use them)
    mmlu_shots: int = 5                  # 5 = primary protocol, 0 = robustness
    humaneval_mode: str = "generate"     # "generate" (main) | "fix" (distribution shift)
    humaneval_timeout: float = 8.0       # per-problem sandbox wall-clock limit (s)
    judge_model: str = "gpt-4o-mini"     # MT-Bench judge (OpenAI-compatible model id)
    method: str = "bg_lora"
    seed: int = 0
    # adaptation
    target_preset: str = "attn_qv"
    n_blocks: int = 0
    rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    use_rslora: bool = True
    # optimizer + schedule (decoupled AdamW; adam_lr is the peak LR, annealed by the schedule)
    adam_lr: float = 2e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0
    steps: int = 300
    schedule_kind: str = "cosine"
    warmup: int = 20
    final_frac: float = 0.0
    gradient_clipping: float = 0.0
    # eval / checkpoints
    eval_every: int = 25
    eval_steps: list = None              # explicit list overrides eval_every
    periodic_gen_eval: bool = True       # False -> skip the mid-training task metric
                                         # (val_ce still drives early stopping; the FINAL
                                         # full-test eval still runs). Big cost saver on
                                         # generation tasks; no effect on reported numbers.
    monitor: str = "val_ce"
    monitor_mode: str = "max"            # "min" for val_ce (lower=better); "max" for a task
                                         # metric like accuracy (higher=better). MUST match
                                         # `monitor` or you would select the WORST checkpoint.
    early_stop: bool = True
    patience: int = 4
    robust_eval: bool = False            # GLUE prompt-robustness: after the normal final
                                         # eval, re-evaluate the SAME weights across K prompt
                                         # templates and record robust_mean/std/min. Opt-in;
                                         # no effect on final_test_metric.
    judge_later: bool = False            # MT-Bench DECOUPLED mode: generate answers and save
                                         # them, but DO NOT call the judge now (score = NaN).
                                         # Judge offline later with scripts/judge_mtbench.py.
    ckpt_gen_examples: int = 100
    save_checkpoints: bool = False       # compact adapter state at each checkpoint
    restore_best: bool = True            # True -> restore best-monitor weights (GLUE);
                                         # False -> keep LAST-step weights (generative,
                                         # fixed-budget protocol). Set per task by
                                         # config.eval_profile (SELECT_GLUE/SELECT_GEN).
    # data
    train_source: str = "openai/gsm8k"
    dataset_revision: str = None
    preprocessing_version: str = "v1"
    train_examples: int = 2000
    val_examples: int = 200
    train_batch_size: int = 8
    grad_accum: int = 1
    val_batch_size: int = 8
    max_len: int = 512
    eval_examples: int = 500
    eval_batch_size: int = 8
    max_new_tokens: int = 256
    # infra
    dtype: str = "bf16"
    device: str = "cuda"
    attn_implementation: str = None
    gradient_checkpointing: bool = False
    offline: bool = False
    overwrite: bool = False              # False -> skip if this run's summary.json exists
    log_every: int = 25
    method_overrides: dict = field(default_factory=dict)
    # bg_lora coefficient estimator: "kalman" (default) | "adam" (preconditioner analysis).
    coeff_optimizer: str = "kalman"
    adam_step_cap: float = 1.0           # bg adam coeff stability: |m| <= 1/cap

    def eval_step_list(self):
        if self.eval_steps:
            return sorted(set(self.eval_steps))
        return sorted(set(list(range(0, self.steps, self.eval_every)) + [self.steps - 1]))

    def spec(self) -> MethodSpec:
        base = dict(name=self.method, r=self.rank, lora_alpha=self.lora_alpha,
                    use_rslora=self.use_rslora,
                    adam_lr=self.adam_lr, adam_beta1=self.adam_beta1,
                    adam_beta2=self.adam_beta2, adam_eps=self.adam_eps,
                    adam_weight_decay=self.adam_weight_decay,
                    coeff_optimizer=self.coeff_optimizer, adam_step_cap=self.adam_step_cap)
        base.update(METHODS[self.method].default_spec_kwargs())
        base.update(self.method_overrides)                        # method_overrides wins
        return MethodSpec(**base)


def param_accounting(method: str, refs, r: int, total_model_params: int) -> dict:
    """The efficiency story, recorded per run: how many parameters GRAB-LoRA actually
    learns, and how its (optimizer-free) memory compares to full fine-tuning.

    GRAB-LoRA learns rank-r factors of the adapted matrices plus the Kalman filter state
    (m,P,rho) per layer, and uses NO Adam optimizer state. Full fine-tuning would instead
    update every adapted weight and pay two fp32 Adam moments per parameter -- the
    reference this is compared against.
    """
    full_targets = sum(rf.d_in * rf.d_out for rf in refs.values())     # full-rank on targets
    lora = sum(r * (rf.d_in + rf.d_out) for rf in refs.values())       # A:(r,d_in)+B:(d_out,r)
    trainable = lora + 3 * r * len(refs)                               # +m,P,rho per layer
    B = 4                                                              # fp32 Adam moment bytes
    mib = lambda n: round(n / (1024 ** 2), 3)
    return {
        "trainable_params": int(trainable),
        "trainable_pct_of_model": round(100.0 * trainable / max(total_model_params, 1), 4),
        "whole_model_params": int(total_model_params),
        "adapted_full_rank_params": int(full_targets),                 # full-FT reference
        "param_reduction_vs_full_rank": round(full_targets / max(trainable, 1), 1),
        "grad_mib": mib(trainable * 2),                                # bf16 grads on trainable
        "optimizer_state_mib": 0.0,                                    # GRAB-LoRA uses no optimizer
        "uses_optimizer": False,
        "full_ft_optimizer_state_mib": mib(2 * total_model_params * B),
        "optimizer_state_saving_vs_full_ft": None,
    }


def _config_dict(cfg: RunConfig, spec, refs, exp_id, acct=None):
    d = {
        "experiment": {"experiment_id": exp_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "seed": cfg.seed, "deterministic": True, "task": cfg.task,
                       "mmlu_shots": cfg.mmlu_shots, "humaneval_mode": cfg.humaneval_mode,
                       "judge_model": cfg.judge_model,
                       "periodic_gen_eval": cfg.periodic_gen_eval},
        "model": {"model_id": cfg.model_id, "revision": cfg.revision, "dtype": cfg.dtype,
                  "quantization": None, "tokenizer_id": cfg.model_id,
                  "max_seq_length": cfg.max_len,
                  "gradient_checkpointing": cfg.gradient_checkpointing,
                  "attention_implementation": cfg.attn_implementation},
        "dataset": {"name": cfg.train_source, "revision": cfg.dataset_revision,
                    "train_split": "train", "validation_split": "train[-val]",
                    "test_split": "test", "prompt_template": cfg.task,
                    "preprocessing_version": cfg.preprocessing_version},
        "training": {"max_steps": cfg.steps, "eval_steps": cfg.eval_step_list(),
                     "save_steps": cfg.eval_step_list() if cfg.save_checkpoints else [],
                     "batch_size": cfg.train_batch_size, "gradient_accumulation": cfg.grad_accum,
                     "effective_batch_size": cfg.train_batch_size * cfg.grad_accum,
                     "optimizer": "adamw", "learning_rate": spec.adam_lr,
                     "scheduler": cfg.schedule_kind, "warmup_steps": cfg.warmup,
                     "weight_decay": cfg.adam_weight_decay,
                     "gradient_clipping": cfg.gradient_clipping},
        "adapter": {"method": cfg.method, "rank": cfg.rank, "alpha": cfg.lora_alpha,
                    "dropout": cfg.lora_dropout, "target_modules": cfg.target_preset,
                    **(acct or {})},
    }
    if cfg.method.startswith("bg_lora"):
        d["bg_lora"] = {"evidence_n": spec.evidence_n, "evidence_bs": spec.evidence_bs,
                        "refresh_interval": spec.refresh_interval,
                        "kappa_u": spec.kappa_u, "kappa_v": spec.kappa_v,
                        "alpha_lambda": spec.lambda_shrink, "beta_estimator": "sample_var",
                        "q0": spec.q0, "q1": spec.q1, "alignment_method": "hungarian",
                        "coefficient_dynamics": spec.coeff_optimizer, "rediag": spec.rediag}
    return d


def run_task(cfg: RunConfig, base_out: str = "runs", commit_fn=None):
    """`commit_fn`, if given, is called after every checkpoint -- for launchers that want
    to persist partial progress so it survives a later crash/timeout."""
    import json as _json
    from ..storage.run_writer import run_dir as _run_dir
    # SKIP-IF-EXISTS: if this exact run already has a summary.json, don't redo it.
    task = get_task(cfg)                              # cheap (no model)
    ku, q0 = cfg.method_overrides.get("kappa_u"), cfg.method_overrides.get("q0")
    variant = f"k{ku:g}_q{q0:g}" if (ku is not None and q0 is not None) else ""
    rd = _run_dir(base_out, task.label, cfg.model_id, cfg.method, cfg.seed, variant)
    sp = os.path.join(rd, "summary.json")
    if not cfg.overwrite and os.path.exists(sp):
        try:
            s = _json.load(open(sp))
            s["skipped"] = True
            return s
        except Exception:
            pass                                     # unreadable -> fall through and rerun

    torch.manual_seed(cfg.seed)
    exp_id = f"{cfg.task}-{cfg.method}-s{cfg.seed}-{content_hash([cfg.method_overrides, cfg.adam_lr, cfg.steps])}"
    model, tok = load_model_and_tokenizer(
        cfg.model_id, cfg.dtype, cfg.device, revision=cfg.revision,
        attn_implementation=cfg.attn_implementation,
        gradient_checkpointing=cfg.gradient_checkpointing)
    target_mods, refs = find_targets(model, cfg.target_preset, cfg.n_blocks)
    freeze_and_arm(model, target_mods)
    keys = list(refs)
    acct = param_accounting(cfg.method, refs, cfg.rank,
                            sum(p.numel() for p in model.parameters()))

    train_pairs, val_pairs, test_rows = task.load(tok)
    # Seed-dependent training ORDER. Shuffling the training pairs per seed gives each
    # seed a different minibatch trajectory AND a different init batch, so repeated seeds
    # produce genuine run-to-run variance. Uses a PRIVATE RNG so torch's global stream
    # (adapter init, bg evidence draws) stays reproducible for a given seed.
    import random as _random
    _random.Random(cfg.seed).shuffle(train_pairs)
    # Encoder-decoder (T5) needs split prompt/target batches + first-decoder-step eval;
    # causal (Qwen/Llama) uses the concatenated causal batches. One flag drives both.
    _seq2seq = is_seq2seq(cfg.model_id)
    try:
        cfg._is_seq2seq = _seq2seq          # read by GLUETask.evaluate for the eval path
    except Exception:
        pass
    _tok_fn = tokenize_pairs_seq2seq if _seq2seq else tokenize_pairs
    train_batches = _tok_fn(train_pairs, tok, cfg.train_batch_size,
                            max(cfg.steps, 1), cfg.max_len, cfg.device)
    n_vb = max(1, min(8, len(val_pairs) // max(cfg.val_batch_size, 1)))
    val_batches = _tok_fn(val_pairs, tok, cfg.val_batch_size, n_vb,
                          cfg.max_len, cfg.device)

    spec = cfg.spec()
    task_label = task.label                          # "gsm8k" or "glue_sst2"
    # sweep cells share (task, model, method, seed); the kappa/q variant keeps their run
    # directories distinct instead of clobbering one another.
    writer = RunWriter(base_out, task_label, cfg.model_id, cfg.method, cfg.seed, variant)
    writer.config(_config_dict(cfg, spec, refs, exp_id, acct))
    writer.environment({"experiment_id": exp_id})
    writer.dataset_manifest({
        "task": task_label, "train_source": cfg.train_source, "metric": task.metric,
        "preprocessing_version": cfg.preprocessing_version,
        "n_train": len(train_pairs), "n_val": len(val_pairs), "n_test": len(test_rows),
        "train_hash": content_hash(p[0] for p in train_pairs)})

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    eval_wall = [0.0]
    start_sub = [None]

    def _cov_mean(recs):
        vals = [r["coverage"] for r in recs
                if isinstance(r.get("coverage"), float) and r["coverage"] == r["coverage"]]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def checkpoint_fn(step, method, refs, keys, Gfull, ctx=None):
        # One checkpoint failing (e.g. an OOM in generation) must not abort the run:
        # record what we can, free memory, persist, and let training continue.
        ctx = ctx or {}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()      # isolate the EVAL-phase peak
        t0 = time.perf_counter()
        vce, acc, preds, err = float("nan"), float("nan"), [], None
        try:
            vce = val_ce(model, val_batches)               # ALWAYS: drives early stopping
            if cfg.periodic_gen_eval:                       # optional mid-training task metric
                acc, _mname, preds = task.evaluate(model, tok, test_rows, cfg.ckpt_gen_examples)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
            _free_cuda()
        eval_dt = time.perf_counter() - t0
        eval_wall[0] += eval_dt
        eval_peak = _peak_mem()
        try:
            if start_sub[0] is None:
                start_sub[0] = geo.subspace_stack(method, keys)
            recs = geo.layer_records(method, refs, keys, Gfull, start_sub[0], cfg.rank)
            writer.geometry(step, recs)
            cov = _cov_mean(recs)
        except Exception as e:
            cov = float("nan")
            err = (err or "") + f" | geometry: {type(e).__name__}: {e}"[:200]
        row = {"step": step, "val_ce": vce, "accuracy": acc, "coverage": cov}
        if err:
            row["error"] = err
        writer.eval(row)
        tsec = float(ctx.get("train_seconds", 0.0))
        toks = int(ctx.get("tokens_seen", 0))
        writer.resource({"step": step,
                         "train_seconds": round(tsec, 2),
                         "train_peak_gpu_mib": int(ctx.get("train_peak_gpu_mib", 0)),
                         "eval_seconds_total": round(eval_wall[0], 2),
                         "eval_seconds_step": round(eval_dt, 2),
                         "eval_peak_gpu_mib": eval_peak,
                         "peak_gpu_memory_mib": max(eval_peak, int(ctx.get("train_peak_gpu_mib", 0))),
                         "tokens_per_second": round(toks / max(tsec, 1e-6), 1)})
        if cfg.save_checkpoints:
            try:
                writer.checkpoint(step, geo.compact_state(method, keys))
            except Exception:
                pass
        if preds and step == cfg.eval_step_list()[-1]:
            writer.predictions(f"checkpoint_{step:06d}", preds)
        # RAW z-evidence: flush the per-refresh coordinate samples bg_lora buffered since
        # the last checkpoint. No-op for other methods.
        if hasattr(method, "pop_z_evidence"):
            try:
                writer.z_evidence(method.pop_z_evidence())
            except Exception:
                pass
        if hasattr(method, "pop_precond_trace"):  # per-atom preconditioner trace (opt-in)
            try:
                _flush_precond(writer.dir, method.pop_precond_trace())
            except Exception:
                pass
        if commit_fn is not None:                 # persist partial progress every checkpoint
            try:
                commit_fn()
            except Exception:
                pass
        return {"val_ce": vce, "accuracy": acc}

    schedule = make_schedule(cfg.schedule_kind, cfg.steps, cfg.warmup, cfg.final_frac)
    method = build(cfg.method, spec)
    train_start = time.perf_counter()
    tr_rows, ev_rows, best = train_lm(
        model, target_mods, refs, method, train_batches, cfg.adam_lr, cfg.steps,
        schedule=schedule, checkpoint_fn=checkpoint_fn,
        eval_steps=cfg.eval_step_list(), early_stop=cfg.early_stop,
        patience=cfg.patience, monitor=cfg.monitor, mode=cfg.monitor_mode, log=cfg.log_every,
        save_subspaces=(cfg.method == "full"), save_dir=writer.dir,
        restore_best=cfg.restore_best,   # GLUE -> best ckpt; generative -> last weights
        seed=cfg.seed)          # ONE seed governs global RNG + bg evidence draws
    # training-only wall (excludes eval), from the loop's own accounting
    train_wall = tr_rows[-1]["train_seconds"] if tr_rows else (time.perf_counter() - train_start)
    for r in tr_rows:
        writer.train(r)
    # Final flush of any raw z-evidence buffered after the last checkpoint. No-op for
    # non-bg methods.
    if hasattr(method, "pop_z_evidence"):
        try:
            writer.z_evidence(method.pop_z_evidence())
        except Exception:
            pass
    if hasattr(method, "pop_precond_trace"):
        try:
            _flush_precond(writer.dir, method.pop_precond_trace())
        except Exception:
            pass

    # full-set final eval at the restored best weights. GUARDED: the final eval is the
    # heaviest step (MT-Bench generates 80x2 long completions on a 7B model, then calls
    # the judge). An OOM or a judge/network failure here must NOT kill the run before
    # summary.json is written; on failure, record NaN + the error and still write every
    # summary field below.
    fe0 = time.perf_counter()
    final_eval_error = None
    try:
        final_acc, metric_name, final_preds = task.evaluate(model, tok, test_rows)
    except Exception as e:
        final_acc, metric_name, final_preds = float("nan"), task.metric, []
        final_eval_error = f"final_eval: {type(e).__name__}: {e}"[:300]
        _free_cuda()
    try:
        final_eval_loss = val_ce(model, val_batches)
    except Exception as e:
        final_eval_loss = float("nan")
        final_eval_error = (final_eval_error or "") + f" | val_ce: {type(e).__name__}: {e}"[:200]
    eval_wall[0] += time.perf_counter() - fe0
    if final_preds:
        writer.predictions("final", final_preds)
    # DECOUPLED MT-BENCH: when judging is deferred, also save the raw generated answers
    # under a stable name so scripts/judge_mtbench.py can find and grade them later.
    if getattr(cfg, "judge_later", False) and task_label == "mtbench" and final_preds:
        try:
            writer.predictions("mtbench_raw", final_preds)
        except Exception:
            pass

    # PROMPT ROBUSTNESS (opt-in via --robust-eval). GLUE only. Re-evaluate the SAME
    # restored-best weights across K prompt templates (0 = the training template,
    # 1..K-1 = unseen paraphrases). Records robust_mean/std/min/max/range + per-template
    # metrics. Does NOT touch final_test_metric.
    robust = None
    if getattr(cfg, "robust_eval", False) and hasattr(task, "evaluate_robust"):
        try:
            rsum, _rname = task.evaluate_robust(model, tok, test_rows)
            robust = rsum
            writer.eval({"step": "robust", **rsum})   # also stream to eval_metrics.jsonl
        except Exception as e:
            robust = {"robust_error": f"{type(e).__name__}: {e}"[:200]}

    best_ckpt_acc = max((r["accuracy"] for r in ev_rows), default=float("nan"))
    total_tokens = tr_rows[-1]["tokens_seen"] if tr_rows else 0
    final_cov = tr_rows[-1]["coverage"] if tr_rows else float("nan")
    summary = {
        "experiment_id": exp_id, "task": task_label, "metric_name": metric_name,
        "model": cfg.model_id, "method": cfg.method, "seed": cfg.seed,
        "best_step": best["best_step"], "final_step": len(tr_rows) - 1,
        "diverged": bool(best.get("diverged", False)),   # True -> hit non-finite loss/update
        "final_eval_error": final_eval_error,   # non-null -> final eval OOM'd or judge failed
        "best_validation_metric": best["best_val"], "best_ckpt_accuracy": best_ckpt_acc,
        "final_test_metric": final_acc, "final_eval_loss": final_eval_loss,
        # prompt-robustness; None unless --robust-eval on a GLUE task
        "robust_mean": (robust or {}).get("robust_mean"),
        "robust_std": (robust or {}).get("robust_std"),
        "robust_min": (robust or {}).get("robust_min"),
        "robust_max": (robust or {}).get("robust_max"),
        "robust_range": (robust or {}).get("robust_range"),
        "robust_per_template": (robust or {}).get("robust_per_template"),
        "peak_gpu_memory_mib": _peak_mem(),
        "training_wall_seconds": round(train_wall, 1),
        "evaluation_wall_seconds": round(eval_wall[0], 1),
        "total_tokens": int(total_tokens),
        "tokens_per_second": round(total_tokens / max(train_wall, 1e-6), 1),
        "kappa": cfg.method_overrides.get("kappa_u"),
        "q": cfg.method_overrides.get("q0"),
        # --- parameter / optimizer-state efficiency (the LoRA-vs-full-FT story) ---
        "trainable_params": acct["trainable_params"],
        "trainable_pct_of_model": acct["trainable_pct_of_model"],
        "optimizer_state_mib": acct["optimizer_state_mib"],
        "param_reduction_vs_full_rank": acct["param_reduction_vs_full_rank"],
        "optimizer_state_saving_vs_full_ft": acct["optimizer_state_saving_vs_full_ft"],
        "whole_model_params": acct["whole_model_params"],
        # convenience aliases for downstream sweep/plot code
        "accuracy": final_acc, "final_coverage": final_cov,
        "best_val_ce": best["best_val"], "run_dir": writer.dir,
        "adam_lr": spec.adam_lr,
        "coeff_optimizer": spec.coeff_optimizer,  # bg only: "kalman" | "adam"
        "restore_best": cfg.restore_best,         # True -> best ckpt; False -> last weights
    }
    writer.summary(summary)
    writer.close()
    return summary


def _peak_mem():
    import torch
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return 0