from __future__ import annotations

import torch

from ..experiments.math_grader import grade as _grade, extract_boxed_answer
from ..data.math import format_prompt


@torch.no_grad()
def evaluate_math(model, tok, rows, max_new_tokens=512, eval_batch_size=8, device="cuda"):
    """rows: [{problem, solution}]. Returns (accuracy, predictions) where each prediction is
    {problem, gold, pred_text, pred_ans, correct}."""
    prev_side = tok.padding_side
    tok.padding_side = "left"                       # completion starts at the prompt's end
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    preds, n_correct = [], 0
    for i in range(0, len(rows), eval_batch_size):
        chunk = rows[i:i + eval_batch_size]
        prompts = [format_prompt(r["problem"]) for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]     # strip the prompt tokens
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for r, text in zip(chunk, texts):
            ok = _grade(text, r["solution"])
            n_correct += int(ok)
            preds.append({"problem": r["problem"], "gold": extract_boxed_answer(r["solution"]),
                          "pred_text": text, "pred_ans": extract_boxed_answer(text),
                          "correct": bool(ok)})
    tok.padding_side = prev_side
    acc = n_correct / max(len(rows), 1)
    return acc, preds