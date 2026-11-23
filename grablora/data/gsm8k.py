from __future__ import annotations

import re

PROMPT = ("Below is a grade-school math problem. Solve it step by step and give the "
          "final answer after '####'.\n\nQuestion: {q}\nAnswer:")

SYNTHETIC_GSM8K = [
    {"question": "Tom has 3 apples and buys 2 more. How many apples does he have?",
     "answer": "He starts with 3 and buys 2, so 3 + 2 = 5.\n#### 5", "final": "5"},
    {"question": "A box holds 4 pens. How many pens are in 3 boxes?",
     "answer": "Each box has 4 pens, so 3 x 4 = 12.\n#### 12", "final": "12"},
    {"question": "Sara had 10 candies and ate 4. How many are left?",
     "answer": "10 - 4 = 6 candies remain.\n#### 6", "final": "6"},
    {"question": "There are 6 rows of 5 chairs. How many chairs total?",
     "answer": "6 x 5 = 30 chairs.\n#### 30", "final": "30"},
]


def _final_from_answer(ans: str) -> str:
    if "####" in ans:
        return ans.split("####")[-1].strip().replace(",", "")
    return extract_answer(ans)


def load_gsm8k(split: str, n: int = 0, source: str = "openai/gsm8k", offline: bool = False):
    if offline:
        rows = list(SYNTHETIC_GSM8K)
        return rows[:n] if n else rows
    from datasets import load_dataset
    cfg = "main" if "gsm8k" in source else None
    ds = load_dataset(source, cfg, split=split) if cfg else load_dataset(source, split=split)
    rows = []
    for ex in ds:
        q = ex.get("question") or ex.get("query") or ex.get("problem")
        a = ex.get("answer") or ex.get("response") or ex.get("solution") or ""
        rows.append({"question": q, "answer": a, "final": _final_from_answer(a)})
        if n and len(rows) >= n:
            break
    return rows


def format_prompt(q: str) -> str:
    return PROMPT.format(q=q)


def extract_answer(text: str) -> str:
    if "####" in text:
        tail = text.split("####")[-1]
        m = re.search(r"-?\d[\d,]*\.?\d*", tail)
        if m:
            return m.group(0).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else ""


def tokenize_train_batches(rows, tok, batch_size, n_batches, max_len=512, device="cpu"):
    import torch
    tok.padding_side = "right"                    # right-pad for teacher forcing
    batches = []
    i = 0
    for _ in range(n_batches):
        chunk = [rows[(i + j) % len(rows)] for j in range(batch_size)]
        i += batch_size
        input_ids, labels = [], []
        for ex in chunk:
            p = format_prompt(ex["question"])
            full = p + " " + ex["answer"] + tok.eos_token
            pid = tok(p, add_special_tokens=False)["input_ids"]
            fid = tok(full, add_special_tokens=False)["input_ids"][:max_len]
            lab = [-100] * min(len(pid), len(fid)) + fid[len(pid):]
            lab = lab[:len(fid)]
            input_ids.append(fid)
            labels.append(lab)
        m = max(len(x) for x in input_ids)
        pad = tok.pad_token_id
        att = [[1] * len(x) + [0] * (m - len(x)) for x in input_ids]
        input_ids = [x + [pad] * (m - len(x)) for x in input_ids]
        labels = [x + [-100] * (m - len(x)) for x in labels]
        batches.append(dict(
            input_ids=torch.tensor(input_ids, device=device),
            attention_mask=torch.tensor(att, device=device),
            labels=torch.tensor(labels, device=device)))
    tok.padding_side = "left"
    return batches