from __future__ import annotations

import torch

from .data.gsm8k import load_gsm8k, format_prompt as gsm_prompt
from .data.glue import (load_glue, format_glue_prompt, VERBALIZERS,
                        metric_name as glue_metric_name)
from .data.alpaca import load_alpaca, alpaca_pairs
from .data.mmlu import load_mmlu_split, build_fewshot_index, format_mmlu_prompt
from .data.codefeedback import load_codefeedback, code_pairs
from .data.wizardlm import load_wizardlm, wizardlm_pairs
from .data.humaneval import load_humaneval
from .data.mtbench import load_mtbench
from .eval.gsm8k_eval import evaluate_gsm8k
from .eval.glue_eval import evaluate_glue
from .eval.glue_eval import evaluate_glue_seq2seq        # T5 path (additive)
from .eval.mmlu_eval import evaluate_mmlu
from .eval.humaneval_eval import evaluate_humaneval
from .eval.mtbench_eval import evaluate_mtbench
from .data.math import load_math, format_prompt as math_prompt
from .eval.math_eval import evaluate_math


def tokenize_pairs(pairs, tok, bs, n_batches, max_len, device):
    tok.padding_side = "right"
    batches, i = [], 0
    for _ in range(n_batches):
        chunk = [pairs[(i + j) % len(pairs)] for j in range(bs)]
        i += bs
        ids, labels = [], []
        for prompt, target in chunk:
            pid = tok(prompt, add_special_tokens=False)["input_ids"]
            fid = tok(prompt + target, add_special_tokens=False)["input_ids"]
            fid = (fid + [tok.eos_token_id])[:max_len]
            lab = ([-100] * min(len(pid), len(fid)) + fid[len(pid):])[:len(fid)]
            ids.append(fid); labels.append(lab)
        m = max(len(x) for x in ids); pad = tok.pad_token_id
        att = [[1] * len(x) + [0] * (m - len(x)) for x in ids]
        ids = [x + [pad] * (m - len(x)) for x in ids]
        labels = [x + [-100] * (m - len(x)) for x in labels]
        batches.append(dict(input_ids=torch.tensor(ids, device=device),
                            attention_mask=torch.tensor(att, device=device),
                            labels=torch.tensor(labels, device=device)))
    tok.padding_side = "left"
    return batches


def tokenize_pairs_seq2seq(pairs, tok, bs, n_batches, max_len, device):
    import torch as _torch
    prev_side = tok.padding_side
    tok.padding_side = "right"
    batches, i = [], 0
    for _ in range(n_batches):
        chunk = [pairs[(i + j) % len(pairs)] for j in range(bs)]
        i += bs
        prompts = [p for p, _ in chunk]
        targets = [t for _, t in chunk]
        enc = tok(prompts, add_special_tokens=True, truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        tgt_ids = [tok(t, add_special_tokens=False, truncation=True,
                       max_length=max_len - 1)["input_ids"] + [tok.eos_token_id]
                   for t in targets]
        L = max(len(x) for x in tgt_ids)
        labels = [x + [-100] * (L - len(x)) for x in tgt_ids]
        batches.append(dict(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            labels=_torch.tensor(labels, device=device)))
    tok.padding_side = prev_side
    return batches


class GSM8KTask:
    label = "gsm8k"
    metric = "accuracy"

    def __init__(self, cfg):
        self.cfg = cfg

    def load(self, tok):
        c = self.cfg
        allt = load_gsm8k("train", c.train_examples + c.val_examples, c.train_source, c.offline)
        train, val = allt[:c.train_examples] or allt, allt[c.train_examples:] or allt
        test = load_gsm8k("test", c.eval_examples, "openai/gsm8k", c.offline)
        pairs = lambda rows: [(gsm_prompt(r["question"]), " " + r["answer"]) for r in rows]
        return pairs(train), pairs(val), test

    def evaluate(self, model, tok, rows, n=None):
        c = self.cfg
        rows = rows[:n] if n else rows
        acc, preds = evaluate_gsm8k(model, tok, rows, c.max_new_tokens,
                                    c.eval_batch_size, c.device)
        return acc, "accuracy", preds

class MATHTask:
    label = "math"
    metric = "accuracy"

    def __init__(self, cfg):
        self.cfg = cfg

    def load(self, tok):
        c = self.cfg
        allt = load_math("train", c.train_examples + c.val_examples, c.train_source, c.offline)
        train, val = allt[:c.train_examples] or allt, allt[c.train_examples:] or allt
        test = load_math("test", c.eval_examples, c.train_source, c.offline)
        pairs = lambda rows: [(math_prompt(r["problem"]), " " + r["solution"]) for r in rows]
        return pairs(train), pairs(val), test

    def evaluate(self, model, tok, rows, n=None):
        c = self.cfg
        rows = rows[:n] if n else rows
        acc, preds = evaluate_math(model, tok, rows, c.max_new_tokens,
                                   c.eval_batch_size, c.device)
        return acc, "accuracy", preds
    
class GLUETask:
    def __init__(self, cfg):
        self.cfg = cfg
        self.subset = cfg.glue_subset
        self.label = f"glue_{self.subset}"
        self.metric = glue_metric_name(self.subset)

    def load(self, tok):
        c = self.cfg
        allt = load_glue(self.subset, "train", c.train_examples + c.val_examples, c.offline)
        train, val = allt[:c.train_examples] or allt, allt[c.train_examples:] or allt
        test = load_glue(self.subset, "test", c.eval_examples, c.offline)   # GLUE validation
        pairs = lambda rows: [(format_glue_prompt(self.subset, r),
                               " " + VERBALIZERS[r["label01"]]) for r in rows]
        return pairs(train), pairs(val), test

    def evaluate(self, model, tok, rows, n=None):
        rows = rows[:n] if n else rows
        # T5 (encoder-decoder) uses first-decoder-step yes/no scoring; causal uses
        # last-prompt-token logits. Flagged on cfg by run_task via is_seq2seq(model_id).
        if getattr(self.cfg, "_is_seq2seq", False):
            val, name, preds = evaluate_glue_seq2seq(model, tok, rows, self.subset,
                                                     self.cfg.eval_batch_size, self.cfg.device)
        else:
            val, name, preds = evaluate_glue(model, tok, rows, self.subset,
                                             self.cfg.eval_batch_size, self.cfg.device)
        return val, name, preds

    def evaluate_robust(self, model, tok, rows, n=None):
        from .eval.glue_eval import evaluate_glue_robust
        from .data.glue import n_templates as _n_templates
        rows = rows[:n] if n else rows
        seq2seq = bool(getattr(self.cfg, "_is_seq2seq", False))
        k = _n_templates(self.subset)
        return evaluate_glue_robust(model, tok, rows, self.subset,
                                    self.cfg.eval_batch_size, self.cfg.device,
                                    n_templates=k, seq2seq=seq2seq)


class MMLUTask:
    metric = "accuracy"

    def __init__(self, cfg):
        self.cfg = cfg
        self.n_shots = int(getattr(cfg, "mmlu_shots", 5))
        self.label = f"mmlu_{self.n_shots}shot"

    def load(self, tok):
        c = self.cfg
        src = c.train_source if c.train_source and "gsm8k" not in c.train_source else "tatsu-lab/alpaca"
        allt = load_alpaca("train", c.train_examples + c.val_examples, src, c.offline)
        train, val = allt[:c.train_examples] or allt, allt[c.train_examples:] or allt
        dev = load_mmlu_split("dev", 0, "cais/mmlu", c.offline)
        idx = build_fewshot_index(dev)
        test_raw = load_mmlu_split("test", c.eval_examples, "cais/mmlu", c.offline)
        test = [{"prompt": format_mmlu_prompt(r, self.n_shots, idx),
                 "gold": r["answer"], "subject": r["subject"]} for r in test_raw]
        return alpaca_pairs(train), alpaca_pairs(val), test

    def evaluate(self, model, tok, rows, n=None):
        rows = rows[:n] if n else rows
        return evaluate_mmlu(model, tok, rows, self.cfg.eval_batch_size, self.cfg.device)


class HumanEvalTask:

    metric = "pass@1"

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = getattr(cfg, "humaneval_mode", "generate")
        self.label = "humaneval" if self.mode == "generate" else "humaneval_fix"

    def load(self, tok):
        c = self.cfg
        src = (c.train_source if c.train_source and "gsm8k" not in c.train_source
               else "m-a-p/CodeFeedback-Filtered-Instruction")
        allt = load_codefeedback("train", c.train_examples + c.val_examples, src, c.offline)
        train, val = allt[:c.train_examples] or allt, allt[c.train_examples:] or allt
        test = load_humaneval(self.mode, c.eval_examples, c.offline)
        return code_pairs(train), code_pairs(val), test

    def evaluate(self, model, tok, rows, n=None):
        c = self.cfg
        rows = rows[:n] if n else rows
        return evaluate_humaneval(model, tok, rows, c.max_new_tokens, c.eval_batch_size,
                                  c.device, getattr(c, "humaneval_timeout", 8.0))


class MTBenchTask:
    label = "mtbench"
    metric = "mt_bench_score"

    def __init__(self, cfg):
        self.cfg = cfg

    def load(self, tok):
        c = self.cfg
        src = (c.train_source if c.train_source and "gsm8k" not in c.train_source
               else "WizardLMTeam/WizardLM_evol_instruct_70k")
        allt = load_wizardlm("train", c.train_examples + c.val_examples, src, c.offline)
        train, val = allt[:c.train_examples] or allt, allt[c.train_examples:] or allt
        test = load_mtbench(c.eval_examples, c.offline)
        return wizardlm_pairs(train), wizardlm_pairs(val), test

    def evaluate(self, model, tok, rows, n=None):
        c = self.cfg
        rows = rows[:n] if n else rows
        return evaluate_mtbench(model, tok, rows, getattr(c, "judge_model", "gpt-4o"),
                                c.max_new_tokens, c.device,
                                judge_later=bool(getattr(c, "judge_later", False)))


_TASKS = {"gsm8k": GSM8KTask, "glue": GLUETask, "mmlu": MMLUTask,
          "humaneval": HumanEvalTask, "mtbench": MTBenchTask,
          "math": MATHTask}


def get_task(cfg):
    return _TASKS.get(cfg.task, GSM8KTask)(cfg)