from __future__ import annotations

import math

import torch

from ..data.glue import format_glue_prompt, metric_name, VERBALIZERS


def _verbalizer_ids(tok):
    return [tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in VERBALIZERS]


@torch.no_grad()
def evaluate_glue(model, tok, rows, subset, batch_size=16, device="cuda"):
    was_cache, was_train = model.config.use_cache, model.training
    model.config.use_cache = False
    model.eval()
    no_id, yes_id = _verbalizer_ids(tok)
    preds, golds = [], []
    try:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            prompts = [format_glue_prompt(subset, r) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits                      # (B, T, V)
            # tok.padding_side == "left" for causal LMs (set in models/loading.py), so
            # real tokens are flush-RIGHT and the last real token is the FINAL column for
            # every row. The old `sum(1)-1` was the RIGHT-padding index and read into the
            # padding for any row shorter than the batch max -> pinned accuracy at ~chance
            # (same bug fixed in mmlu_eval.py).
            step = logits[:, -1, :]                            # (B, V)
            pred = (step[:, yes_id] > step[:, no_id]).long().tolist()
            preds += pred
            golds += [r["label01"] for r in chunk]
    finally:
        model.config.use_cache = was_cache
        model.train(was_train)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    name = metric_name(subset)
    return glue_metric(preds, golds, name), name, [
        {"pred": p, "gold": g} for p, g in zip(preds, golds)]


def glue_metric(preds, golds, name) -> float:
    tp = sum(p == 1 and g == 1 for p, g in zip(preds, golds))
    tn = sum(p == 0 and g == 0 for p, g in zip(preds, golds))
    fp = sum(p == 1 and g == 0 for p, g in zip(preds, golds))
    fn = sum(p == 0 and g == 1 for p, g in zip(preds, golds))
    if name == "accuracy":
        return (tp + tn) / max(len(golds), 1)
    if name == "f1":
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return 2 * prec * rec / max(prec + rec, 1e-9)
    if name == "matthews":
        num = tp * tn - fp * fn
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1.0
        return num / den
    raise ValueError(name)

# ---------------------------------------------------------------- T5 / seq2seq path
@torch.no_grad()
def evaluate_glue_seq2seq(model, tok, rows, subset, batch_size=16, device="cuda"):
    """Encoder-decoder GLUE scoring (T5): prompt -> encoder; read FIRST decoder-step
    logits (decoder input = start token) to compare " yes" vs " no". Same
    {"pred","gold"} output as evaluate_glue, so glue_metric and the collapse detector
    both work unchanged."""
    was_cache, was_train = model.config.use_cache, model.training
    model.config.use_cache = False
    model.eval()
    # verbalizer first-token ids under the T5 tokenizer (spaced form first)
    def _vid(w):
        c = tok(" " + w, add_special_tokens=False)["input_ids"]
        if not c:
            c = tok(w, add_special_tokens=False)["input_ids"]
        return c[0]
    no_id, yes_id = _vid(VERBALIZERS[0]), _vid(VERBALIZERS[1])
    dsid = getattr(model.config, "decoder_start_token_id", None)
    if dsid is None:
        dsid = model.config.pad_token_id
    prev_side = tok.padding_side
    tok.padding_side = "right"
    preds, golds = [], []
    try:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            prompts = [format_glue_prompt(subset, r) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            dec_in = torch.full((len(chunk), 1), dsid, dtype=torch.long, device=device)
            logits = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"],
                           decoder_input_ids=dec_in).logits          # (B, 1, V)
            step = logits[:, 0, :]
            pred = (step[:, yes_id] > step[:, no_id]).long().tolist()
            preds += pred
            golds += [r["label01"] for r in chunk]
    finally:
        model.config.use_cache = was_cache
        model.train(was_train)
        tok.padding_side = prev_side
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    name = metric_name(subset)
    return glue_metric(preds, golds, name), name, [
        {"pred": p, "gold": g} for p, g in zip(preds, golds)]

# ---------------------------------------------------------------- all metrics at once
def glue_metric_all(preds, golds) -> dict:
    """Every standard binary metric from ONE set of predictions. Training optimizes CE
    loss (metric-agnostic), so any metric is a post-hoc function of (pred, gold): you
    can train once and report accuracy / F1 / Matthews together, or recompute a
    different metric later from saved predictions/final.jsonl without retraining.

    Returns accuracy, f1, matthews, precision, recall (+ counts). Binary tasks only
    (the yes/no verbalizer: cola, sst2, mrpc, qnli, rte, qqp, wnli). MNLI (3-way) and
    STS-B (regression) use their own metric functions.
    """
    tp = sum(p == 1 and g == 1 for p, g in zip(preds, golds))
    tn = sum(p == 0 and g == 0 for p, g in zip(preds, golds))
    fp = sum(p == 1 and g == 0 for p, g in zip(preds, golds))
    fn = sum(p == 0 and g == 1 for p, g in zip(preds, golds))
    n = max(len(golds), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1.0
    mcc = (tp * tn - fp * fn) / den
    return {"accuracy": (tp + tn) / n, "f1": f1, "matthews": mcc,
            "precision": prec, "recall": rec, "n": len(golds)}