from __future__ import annotations

import torch

from ..data.gsm8k import format_prompt, extract_answer


@torch.no_grad()
def val_ce(model, val_batches) -> float:
    """Mean cross-entropy over held-out labeled batches -- a cheap generalization
    signal to drive early stopping (no generation needed)."""
    was = model.training
    model.eval()
    tot, n = 0.0, 0
    for b in val_batches:
        tot += float(model(**b).loss)
        n += 1
    model.train(was)
    _free_cuda()
    return tot / max(n, 1)


@torch.no_grad()
def evaluate_gsm8k(model, tok, test_rows, max_new_tokens=256, batch_size=8,
                   device="cuda"):
    was_cache, was_training = model.config.use_cache, model.training
    model.config.use_cache = True
    model.eval()
    correct, preds = 0, []
    try:
        for i in range(0, len(test_rows), batch_size):
            chunk = test_rows[i:i + batch_size]
            prompts = [format_prompt(r["question"]) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            cont = gen[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(cont, skip_special_tokens=True)
            for r, txt in zip(chunk, texts):
                pred = extract_answer(txt)
                ok = (pred != "" and pred == r["final"])
                correct += int(ok)
                preds.append({"question": r["question"], "gold": r["final"],
                              "pred": pred, "correct": bool(ok)})
            del enc, gen, cont
    finally:
        model.config.use_cache = was_cache
        model.train(was_training)               # eval() must not leak into training
        _free_cuda()                             # release the generation KV-cache
    return correct / max(len(test_rows), 1), preds


def _free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()