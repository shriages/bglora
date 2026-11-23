# GRAB-LoRA

Reference implementation of **GRAB-LoRA** (Bayesian Grassmannian LoRA, method key
`bg_lora`): a rank-r LoRA-family fine-tuning method that re-reads the gradient every
step, lets its subspace **rotate** under a Grassmannian prior, and runs a per-atom
**Kalman filter** over the coefficients — proposing the weight update directly and
training it with decoupled AdamW. It is model-free (it operates on weight matrices and
gradients), so the same code runs on any Hugging Face causal LM (Qwen / Llama) or on
T5 (encoder–decoder).

> **Defaults are for a quick, runnable smoke test, not the paper.** The commands below
> run a small model on a small data slice in minutes. The exact model, data scale, step
> counts, learning rates, and `kappa`/`q` values used for the reported results are the
> **overrides listed in the paper** — see [Reproducing the paper](#reproducing-the-paper).

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # optional: makes `import grablora` work from anywhere
```

Requires Python ≥ 3.10 and a CUDA build of PyTorch (`torch>=2.5`). `openai` is only
needed for the MT-Bench judge.

## 2. Credentials (environment variables)

```bash
export HF_TOKEN=hf_xxx           # only for GATED models (Llama). Qwen / T5 are ungated.
export OPENAI_API_KEY=sk-xxx     # only for the MT-Bench task (LLM judge)
export OPENAI_BASE_URL=https://... # optional: an OpenAI-compatible endpoint for the judge
```

Models and datasets are pulled from the Hugging Face Hub on first use and cached under
`HF_HOME`. For a gated model, accept its license on the Hub with the account behind
`HF_TOKEN`.

## 3. Quick start (runnable defaults)

```bash
# one task, one seed, one (kappa, q) cell — small scale, finishes in minutes on one GPU
python gpu_run.py --models qwen1.5b --tasks gsm8k --seeds 1 --kappas 10 --qs 10

# preview what a selection would run without launching anything
python gpu_run.py --tasks glue --seeds 1,2,3 --dry-run
```

Results land under `./runs/{task}/{model}/bg_lora/seed_{seed}__k{kappa}_q{q}/`. Re-running
skips any cell that already has a `summary.json` (pass `--overwrite` to force).

## 4. Multi-GPU deployment (8 × A100 40GB)

`gpu_run.py` runs each cell in its own subprocess pinned to one GPU via
`CUDA_VISIBLE_DEVICES`, keeping all GPUs busy until the queue drains. One run per
process means an OOM or CUDA fault kills only that job, not the sweep.

```bash
python gpu_run.py \
  --models llama8b \
  --tasks gsm8k,humaneval,mmlu,mtbench \
  --seeds 1,2,3 \
  --kappas 0.01,1,10,100 --qs 0.01,1,10,100 \
  --scale medium \
  --gpus 8
```

The per-model batch sizes in `grablora/experiments/config.py` (`MODEL_OVERRIDES`) are
already tuned for a **40GB A100**: 7–8B models use `train_batch_size=4`, gradient
checkpointing, and SDPA attention. If a 7–8B run still OOMs, drop the batch to 2 and
raise `--steps` (grad-accum is a no-op in the loop, so add steps rather than grow the
batch).

### Heterogeneous sweeps (per-experiment overrides)

To run many differently-configured cells across the GPUs at once, pass a JSON manifest —
a list of experiment groups, each overriding only what varies:

```bash
python gpu_run.py --spec experiments.json --models llama8b --seeds 1,2,3 --scale medium --gpus 8
```

```json
[
  {"tasks": "humaneval", "adam_lr": 2e-4, "kappas": "10",  "qs": "10"},
  {"tasks": "mmlu",      "adam_lr": 5e-5, "kappas": "1",   "qs": "1"},
  {"tasks": "gsm8k",     "adam_lr": 1e-4, "kappas": "100", "qs": "0.01"}
]
```

`run_spec.py` runs a flat manifest (a `{"runs": [...]}` object or a bare list of
fully-resolved payloads) through the same dispatcher.

## 5. The method

| key       | what it is                                                        |
|-----------|-------------------------------------------------------------------|
| `bg_lora` | **GRAB-LoRA** — rotating rank-r subspace + per-atom Kalman filter |

GRAB-LoRA sweeps a `kappa` × `q` grid (Grassmann prior strength × coefficient process
noise); pass `--kappas` / `--qs`. Its coefficient estimator is `--coeff-optimizer kalman`
(default) or `adam` (the measured-denominator variant used for the preconditioner
analysis; set `--precond` on `run_spec.py`, or `GRABLORA_PRECOND_TRACE=1`, to log the
per-atom trace). Other bg knobs (`refresh_interval`, `evidence_n`, `evidence_bs`,
`j_max`, `lambda_*`, `rediag`, …) live in `grablora/core.py::MethodSpec`.

## 6. Tasks & datasets

| task key    | train set                              | test set / metric        | backbone in the paper |
|-------------|----------------------------------------|--------------------------|-----------------------|
| `glue_*`    | GLUE subset (cola/sst2/mrpc/qnli/rte/qqp) | GLUE val / acc·matthews | T5-base               |
| `gsm8k`     | MetaMathQA                              | GSM8K / accuracy         | Llama-3.1-8B          |
| `math`      | MATH (Hendrycks)                        | MATH / accuracy          | Llama-3.1-8B          |
| `humaneval` | CodeFeedback-Filtered-Instruction       | HumanEval / pass@1       | Llama-3.1-8B          |
| `mmlu`      | Alpaca                                  | MMLU / accuracy          | Llama-3.1-8B          |
| `mtbench`   | WizardLM Evol-Instruct                  | MT-Bench / judge score   | Llama-3.1-8B          |

Selection keywords for `--tasks`: `all`, `glue`, `instruct` (mmlu+humaneval+mtbench), or
a comma list. GLUE uses the best-dev-checkpoint protocol; generative tasks use the
last-step weights — chosen automatically per task.

`humaneval` **executes model-generated code** in a sandboxed subprocess — run it only
where that is acceptable. `mtbench` calls an OpenAI-compatible judge; for a run with no
outbound internet, add `--judge-later` to generate/save answers now and judge them
offline afterward with `scripts/judge_mtbench.py`.

## 7. Outputs