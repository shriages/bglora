from __future__ import annotations

import os

from datasets import load_dataset


_PROMPT = ("Solve the following math problem step by step. "
           "Put the final answer inside \\boxed{{}}.\n\n"
           "Problem: {q}\n\nSolution:")


def format_prompt(problem: str) -> str:
    return _PROMPT.format(q=(problem or "").strip())


def _load(source: str, split: str):
    repo, _, cfg = source.partition(":")
    if cfg:
        return load_dataset(repo, cfg, split=split)
    try:
        return load_dataset(repo, split=split)
    except (ValueError, FileNotFoundError):
        # some MATH mirrors REQUIRE a config; "all" is the usual one
        return load_dataset(repo, "all", split=split)


def load_math(split, n, source="DigitalLearningGmbH/MATH-lighteval:default", offline=False):
    if offline:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # a MATH-specific default source overrides gsm8k's default that config passes through
    if not source or "gsm8k" in source.lower() or "metamath" in source.lower():
        source = "DigitalLearningGmbH/MATH-lighteval:default"
    ds = _load(source, split)
    rows = []
    for r in ds:
        prob = r.get("problem") or r.get("question") or r.get("Problem") or ""
        sol = r.get("solution") or r.get("answer") or r.get("Solution") or ""
        if not prob:
            continue
        rows.append({"problem": prob, "solution": sol})
        if n and len(rows) >= n:
            break
    return rows