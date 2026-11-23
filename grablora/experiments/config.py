from __future__ import annotations

# Shared defaults (adapter + optimizer + eval), applied to every run.
SHARED = dict(
    # adapter
    rank=8, lora_alpha=16.0, use_rslora=True, lora_dropout=0.0,
    target_preset="attn_qv",              # attn_qv | attn_qkv | attn_all | all_linear
    # optimizer (decoupled AdamW) + schedule. The peak LR is adam_lr; the schedule
    # multiplies it. Per-task LR overrides are set on the CLI or in a spec.
    adam_lr=2e-4, schedule_kind="cosine", warmup=20, final_frac=0.0,
    # bg-lora method knobs (used by bg_lora, via method_overrides)
    refresh_interval=30, j_max=1, evidence_n=4, evidence_bs=8,
    # eval / checkpointing
    eval_every=50, ckpt_gen_examples=100, max_new_tokens=256,
    periodic_gen_eval=True,               # False -> only val_ce mid-training + ONE final eval
    monitor="accuracy", monitor_mode="max", early_stop=False, patience=6, save_checkpoints=False,
    restore_best=True,                    # False -> final eval runs on the LAST-step weights
    # per-task eval knobs (defaults; TASKS[*]["over"] and CLI override these)
    mmlu_shots=5, humaneval_mode="generate", humaneval_timeout=8.0,
    judge_model="gpt-4o-mini",            # MT-Bench judge (OpenAI-compatible; set via --judge-model)
    # batching / dtype
    max_len=512, train_batch_size=8, val_batch_size=8, eval_batch_size=8,
    dtype="bf16", device="cuda",
)

# ---- scale profiles: sample counts + steps -----------------------------------
# "small" is the fast, runnable default. "medium"/"large" are the sizes used for the
# reported results (see the paper).
SCALE = {
    "small":  dict(train_examples=2000,  val_examples=200,  eval_examples=300,  steps=300),
    "medium": dict(train_examples=20000, val_examples=500,  eval_examples=1000, steps=3000),
    "large":  dict(train_examples=30000, val_examples=1000, eval_examples=0,    steps=1500),
}
TRAIN_CAP = 30000                          # hard cap on train size at "large"

# ---- task registry: dataset wiring + official sizes --------------------------
# test_n = official evaluation size (GLUE uses the validation split, since GLUE test
# labels are hidden). off_train = official train size (for "large"). `over` = per-task
# RunConfig overrides. `variant` marks robustness/shift add-ons NOT included by tasks="all".
#
# SST-2 / QNLI train on the FULL official set (T5-base) with enough steps for ~2 epochs
# at the T5-base batch (32, see MODEL_OVERRIDES), evaluated on the full GLUE val split.
TASKS = {
    "gsm8k":     dict(task="gsm8k", train_source="meta-math/MetaMathQA", off_train=TRAIN_CAP, test_n=1319),
    "glue_sst2": dict(task="glue", glue_subset="sst2", off_train=67349,  test_n=872,
                      over=dict(train_examples=67349, steps=4200, max_len=128,
                                eval_examples=872, eval_every=200)),
    "glue_mrpc": dict(task="glue", glue_subset="mrpc", off_train=3668,   test_n=408),
    "glue_cola": dict(task="glue", glue_subset="cola", off_train=8551,   test_n=1043,   ckpt_gen_examples=1043),
    "glue_qnli": dict(task="glue", glue_subset="qnli", off_train=104743, test_n=5463,
                      over=dict(train_examples=104743, steps=6500, max_len=256,
                                eval_examples=5463, eval_every=200)),
    "glue_rte":  dict(task="glue", glue_subset="rte",  off_train=2490,   test_n=277),
    "glue_qqp":  dict(task="glue", glue_subset="qqp",  off_train=363846, test_n=40430),
    # ---- instruction / code / chat suites ----
    # Alpaca -> MMLU (accuracy). Primary = 5-shot; robustness add-on = 0-shot.
    "mmlu":        dict(task="mmlu", train_source="tatsu-lab/alpaca", off_train=TRAIN_CAP,
                        test_n=14042, over=dict(mmlu_shots=5)),
    "mmlu_0shot":  dict(task="mmlu", train_source="tatsu-lab/alpaca", off_train=TRAIN_CAP,
                        test_n=14042, over=dict(mmlu_shots=0), variant=True),
    # CodeFeedback -> HumanEval (pass@1). Shift add-on = debug/refine (humanevalpack fix).
    "humaneval":     dict(task="humaneval", train_source="m-a-p/CodeFeedback-Filtered-Instruction",
                          off_train=TRAIN_CAP, test_n=164,
                          over=dict(max_new_tokens=256, humaneval_mode="generate")),
    "humaneval_fix": dict(task="humaneval", train_source="m-a-p/CodeFeedback-Filtered-Instruction",
                          off_train=TRAIN_CAP, test_n=164,
                          over=dict(max_new_tokens=256, humaneval_mode="fix"), variant=True),
    # WizardLM -> MT-Bench (GPT-judge, 1-10).
    "mtbench":     dict(task="mtbench", train_source="WizardLMTeam/WizardLM_evol_instruct_70k",
                        off_train=TRAIN_CAP, test_n=80,
                        over=dict(max_new_tokens=256, ckpt_gen_examples=8)),
    # ---- MATH (Hendrycks): high-rotation math for the rotation story ----
    "math": dict(task="math", train_source="DigitalLearningGmbH/MATH-lighteval:default",
                 off_train=7500, test_n=500, over=dict(max_new_tokens=512)),
}
VARIANT_TASKS = {k for k, v in TASKS.items() if v.get("variant")}

# ---- model-selection profile, chosen AUTOMATICALLY per task family -----------
#   GLUE (classification): dev-metric selection, cheap periodic eval, report the BEST
#       checkpoint (the standard GLUE protocol).
#   Generative (gsm8k / math / mmlu / humaneval / mtbench): val_ce as a cheap guard only,
#       NO per-checkpoint generation (fast), train the full budget, report the LAST-step
#       weights (fixed-budget protocol).
# CLI overrides still win (applied AFTER this profile in build_payloads).
SELECT_GLUE = dict(monitor="accuracy", monitor_mode="max",
                   periodic_gen_eval=True,  early_stop=False, restore_best=True)
SELECT_GEN  = dict(monitor="val_ce",   monitor_mode="min",
                   periodic_gen_eval=False, early_stop=False, restore_best=False)


def eval_profile(task_key: str) -> dict:
    """GLUE -> best-dev-accuracy; everything else (generative) -> last-weights + val_ce guard."""
    return dict(SELECT_GLUE if TASKS[task_key]["task"] == "glue" else SELECT_GEN)


BG_METHOD = "bg_lora"          # the only method in this release (GRAB-LoRA)
# bg_lora kappa x q grid (Grassmann prior strength x coefficient process noise).
SWEEP = dict(kappas=[0.01, 1.0, 10.0, 100.0], qs=[0.01, 1.0, 10.0, 100.0])

MODELS = ["Qwen/Qwen2.5-1.5B"]                                   # runnable default; override with --models
SEEDS = [1, 2, 3]

# ---- model registry: short names -> HF repo ids ------------------------------
# All causal entries are DENSE, TEXT-ONLY causal LMs (AutoModelForCausalLM). The T5
# family is loaded via AutoModelForSeq2SeqLM (encoder-decoder path).
MODEL_REGISTRY = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B",      # ungated
    "qwen4b":   "Qwen/Qwen3-4B-Base",     # ungated (dense text)
    "qwen7b":   "Qwen/Qwen2.5-7B",        # ungated
    "llama3b":  "meta-llama/Llama-3.2-3B",  # GATED: accept license + set HF_TOKEN
    "llama8b":  "meta-llama/Llama-3.1-8B",  # GATED: accept license + set HF_TOKEN
    # ---- T5 family (encoder-decoder; seq2seq path) ----
    "t5small":  "google-t5/t5-small",     # fast checks (~60M)
    "t5base":   "google-t5/t5-base",      # the GLUE backbone (~220M)
    "t5large":  "google-t5/t5-large",     # ~770M
    "flant5base": "google/flan-t5-base",  # instruction-tuned base
}

# ---- per-model RunConfig overrides -------------------------------------------
# Bigger models get smaller batches / gradient checkpointing so training + generation
# fit one 40GB A100. Applied AFTER the task `over` and BEFORE CLI flags (so --* wins).
# KEYED BY BARE MODEL NAME (the id basename), so the same entry matches an HF id or a
# local path (the lookup tries the full id first, then the basename).
MODEL_OVERRIDES = {
    # T5-base is the GLUE backbone and small (~220M): only LoRA adapters train, base
    # frozen, GLUE sequences short -> batch 32 is memory-cheap and the SST-2/QNLI
    # full-data step counts in TASKS[*]["over"] are sized for exactly this batch.
    "t5-base":  dict(train_batch_size=32, val_batch_size=32, eval_batch_size=32),
    "t5-large": dict(train_batch_size=4, val_batch_size=4, eval_batch_size=4),
    # 7-8B on a 40GB A100: gradient checkpointing trades compute for a big activation-
    # memory cut (effective only with loading.py's use_reentrant=False, since the base is
    # frozen); sdpa attention avoids eager attention buffers; eval_batch_size lowered
    # because the eval-phase forward is the memory spike. If batch 4 still OOMs, drop to 2
    # (grad_accum is a no-op in lm_loop, so raise STEPS rather than the batch for coverage).
    "Llama-3.1-8B": dict(train_batch_size=4, val_batch_size=4, eval_batch_size=2,
                         gradient_checkpointing=True, attn_implementation="sdpa"),
    "Qwen2.5-7B":   dict(train_batch_size=4, val_batch_size=4, eval_batch_size=2,
                         gradient_checkpointing=True, attn_implementation="sdpa"),
    "Llama-3.2-3B": dict(train_batch_size=8, val_batch_size=8, eval_batch_size=8),
    "Qwen3-4B-Base":dict(train_batch_size=8, val_batch_size=8, eval_batch_size=8),
}


def model_overrides_for(model_id: str) -> dict:
    """Per-model RunConfig overrides. Tries the full id, then the basename, so both an
    HF id and a local path hit the same entry."""
    return (MODEL_OVERRIDES.get(model_id)
            or MODEL_OVERRIDES.get(model_id.rstrip("/").rsplit("/", 1)[-1], {}))


# ---- resolvers for the CLI keywords ------------------------------------------
def resolve_tasks(spec: str) -> list:
    s = spec.strip().lower()
    if s in ("", "all"):                       # everything EXCEPT robustness/shift add-ons
        return [k for k in TASKS if k not in VARIANT_TASKS]
    if s == "glue":
        return [k for k in TASKS if k.startswith("glue_")]
    if s == "gsm8k":
        return ["gsm8k"]
    if s == "math":
        return ["math"]
    if s == "mmlu":
        return ["mmlu"]
    if s in ("mmlu_robust", "mmlu_all"):        # primary + 0-shot robustness
        return ["mmlu", "mmlu_0shot"]
    if s in ("code", "humaneval"):
        return ["humaneval"]
    if s in ("humaneval_shift", "code_shift"):  # main + debug/refine distribution shift
        return ["humaneval", "humaneval_fix"]
    if s in ("mtbench", "chat"):
        return ["mtbench"]
    if s == "instruct":                         # the three new train->eval pairs, primary only
        return ["mmlu", "humaneval", "mtbench"]
    return [t.strip() for t in spec.split(",") if t.strip()]


def resolve_models(spec: str) -> list:
    """Map the --models keyword to HF repo ids. Accepts short names (qwen7b, llama8b),
    full repo ids, 'all', or '' -> the default MODELS."""
    s = (spec or "").strip()
    if s == "" or s.lower() == "default":
        return list(MODELS)
    if s.lower() == "all":
        return [MODEL_REGISTRY[k] for k in ("qwen1.5b", "llama3b", "qwen4b", "qwen7b", "llama8b")]
    out = []
    for tok in s.split(","):
        t = tok.strip()
        if t:
            out.append(MODEL_REGISTRY.get(t.lower(), t))     # short name -> id, else literal
    return out


def resolve_methods(spec: str = "", _unused: bool = False) -> list:
    # GRAB-LoRA is the only method in this release; every selection resolves to it.
    return [BG_METHOD]


# ---- the builder: selection -> fully-resolved payloads -----------------------
def build_payloads(models, task_keys, methods, seeds, scale="small",
                   kappas=None, qs=None, overwrite=False, overrides=None) -> list:
    """Expand the selection into one fully-specified RunConfig-override dict per run.
    Every payload carries the COMPLETE setting (SHARED+SCALE+TASK+profile+overrides), so
    the saved config.yaml is self-describing. bg_lora expands over the kappa x q grid."""
    kappas = kappas or SWEEP["kappas"]
    qs = qs or SWEEP["qs"]
    overrides = overrides or {}
    sc = SCALE[scale]
    bg_keys = ("refresh_interval", "j_max", "evidence_n", "evidence_bs")
    out = []
    for model in models:
        for tk in task_keys:
            t = TASKS[tk]
            test_n = t["test_n"]
            eval_examples = (sc["eval_examples"] or test_n) if scale == "large" else min(sc["eval_examples"], test_n)
            train_examples = min(sc["train_examples"], t["off_train"]) if scale == "large" else sc["train_examples"]
            merged = dict(SHARED)
            merged.update(model_id=model, task=t["task"],
                          glue_subset=t.get("glue_subset", "sst2"),
                          train_source=t.get("train_source", "openai/gsm8k"),
                          steps=sc["steps"], train_examples=train_examples,
                          val_examples=sc["val_examples"], eval_examples=eval_examples,
                          overwrite=overwrite)
            merged.update(t.get("over", {}))              # per-task eval knobs
            merged.update(eval_profile(tk))               # GLUE vs generative SELECTION profile (auto)
            merged.update(model_overrides_for(model))     # per-model batch/memory (id OR local path)
            merged.update(overrides)                       # CLI wins over everything
            bg_knobs = {k: merged.pop(k) for k in bg_keys}   # -> method_overrides, not RunConfig
            common = merged                                  # now pure RunConfig kwargs
            for seed in seeds:
                # GRAB-LoRA is the only method; expand over the kappa x q grid.
                for k in kappas:
                    for q in qs:
                        p = dict(common, method=BG_METHOD, seed=seed,
                                 method_overrides=dict(kappa_u=k, kappa_v=k,
                                                       q0=q, q1=q, **bg_knobs))
                        p["_task"], p["_kind"], p["_kappa"], p["_q"] = tk, "bg", k, q
                        out.append(p)
    return out