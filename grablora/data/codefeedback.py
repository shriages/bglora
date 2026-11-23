from __future__ import annotations

_TMPL = ("Below is a programming problem. Write a correct solution.\n\n"
         "### Problem:\n{query}\n\n### Solution:\n")

SYNTHETIC_CODEFEEDBACK = [
    {"query": "Write a function add(a,b) that returns a+b.",
     "answer": "```python\ndef add(a, b):\n    return a + b\n```"},
    {"query": "Write a function inc(x) that returns x+1.",
     "answer": "```python\ndef inc(x):\n    return x + 1\n```"},
] * 8


def format_code_prompt(query: str) -> str:
    return _TMPL.format(query=(query or "").strip())


def load_codefeedback(split: str, n: int = 0,
                      source: str = "m-a-p/CodeFeedback-Filtered-Instruction",
                      offline: bool = False):
    if offline:
        rows = list(SYNTHETIC_CODEFEEDBACK)
        return rows[:n] if n else rows
    from datasets import load_dataset
    ds = load_dataset(source, split="train")
    rows = []
    for ex in ds:
        q = ex.get("query") or ex.get("instruction") or ex.get("prompt")
        a = ex.get("answer") or ex.get("response") or ex.get("output") or ""
        if not (q and a):
            continue
        rows.append({"query": q, "answer": a})
        if n and len(rows) >= n:
            break
    return rows


def code_pairs(rows):
    return [(format_code_prompt(r["query"]), " " + r["answer"].strip()) for r in rows]