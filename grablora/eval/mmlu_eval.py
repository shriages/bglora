from __future__ import annotations

import torch

from ..data.mmlu import LETTERS


def _letter_ids(tok):
    ids = []
    for L in LETTERS:
        t = tok(" " + L, add_special_tokens=False)["input_ids"]
        ids.append(t[0])
    return ids


@torch.no_grad()
def evaluate_mmlu(model, tok, items, batch_size=8, device="cuda"):
    """items: list of {prompt, gold(int 0..3), subject}. Returns (accuracy, name, preds)."""
    was_cache, was_train = model.config.use_cache, model.training
    model.config.use_cache = False
    model.eval()
    tok.padding_side = "left"                       # last real token is at position -1
    lids = torch.tensor(_letter_ids(tok), device=device)
    correct, preds = 0, []
    try:
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            prompts = [c["prompt"] for c in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits                    # (B, T, V)
            # padding_side="left" -> real tokens are flush-right, so the last real
            # token is the FINAL column for every row. (The old sum(1)-1 was the
            # RIGHT-padding index and read the wrong position for every padded row,
            # pinning accuracy at ~chance.)
            step = logits[:, -1, :]                          # (B, V)
            choice = step[:, lids].argmax(dim=1).tolist()   # 0..3
            for c, ch in zip(chunk, choice):
                ok = int(ch == c["gold"])
                correct += ok
                preds.append({"subject": c.get("subject", ""), "gold": c["gold"],
                              "pred": ch, "correct": bool(ok)})
            del enc, logits, step
    finally:
        model.config.use_cache = was_cache
        model.train(was_train)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return correct / max(len(items), 1), "accuracy", preds