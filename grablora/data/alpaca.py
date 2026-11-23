from __future__ import annotations

_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the "
    "request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n")

SYNTHETIC_ALPACA = [
    {"instruction": "Name a primary color.", "input": "", "output": "Blue."},
    {"instruction": "Add the numbers.", "input": "2 and 3", "output": "5."},
    {"instruction": "Give an antonym of hot.", "input": "", "output": "Cold."},
    {"instruction": "Capitalize the word.", "input": "apple", "output": "Apple."},
] * 4


def format_alpaca_prompt(ex: dict) -> str:
    inp = (ex.get("input") or "").strip()
    tmpl = _WITH_INPUT if inp else _NO_INPUT
    return tmpl.format(instruction=(ex.get("instruction") or "").strip(), input=inp)


def load_alpaca(split: str, n: int = 0, source: str = "tatsu-lab/alpaca",
                offline: bool = False):
    if offline:
        rows = list(SYNTHETIC_ALPACA)
        return rows[:n] if n else rows
    from datasets import load_dataset
    ds = load_dataset(source, split="train")
    rows = []
    for ex in ds:
        out = (ex.get("output") or ex.get("response") or "").strip()
        if not out:
            continue
        rows.append({"instruction": ex.get("instruction", ""),
                     "input": ex.get("input", ""), "output": out})
        if n and len(rows) >= n:
            break
    return rows


def alpaca_pairs(rows):

    return [(format_alpaca_prompt(r), " " + r["output"].strip()) for r in rows]