from __future__ import annotations

_FIX_TMPL = (
    "Below is a Python function that contains a bug. Fix the bug and return the "
    "complete corrected function.\n\n"
    "### Buggy code:\n```python\n{decl}{buggy}\n```\n\n"
    "### Corrected function:\n```python\n")

# --- offline fixtures (trivial, so the executor path is exercised with no network) ---
SYNTHETIC_HUMANEVAL = [
    {"task_id": "toy/0",
     "prompt": "def add(a, b):\n    \"\"\"Return a+b.\"\"\"\n",
     "canonical_solution": "    return a + b\n",
     "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
     "entry_point": "add"},
    {"task_id": "toy/1",
     "prompt": "def inc(x):\n    \"\"\"Return x+1.\"\"\"\n",
     "canonical_solution": "    return x + 1\n",
     "test": "def check(candidate):\n    assert candidate(10) == 11\n",
     "entry_point": "inc"},
] * 2

SYNTHETIC_HUMANEVAL_FIX = [
    {"task_id": "toyfix/0",
     "declaration": "def add(a, b):\n",
     "buggy_solution": "    return a - b\n",
     "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
     "entry_point": "add"},
    {"task_id": "toyfix/1",
     "declaration": "def inc(x):\n",
     "buggy_solution": "    return x - 1\n",
     "test": "def check(candidate):\n    assert candidate(10) == 11\n",
     "entry_point": "inc"},
] * 2


def load_humaneval(mode: str = "generate", n: int = 0, offline: bool = False):
    if mode == "fix":
        return _load_fix(n, offline)
    return _load_generate(n, offline)


def _load_generate(n, offline):
    if offline:
        rows = list(SYNTHETIC_HUMANEVAL)
    else:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        rows = [dict(task_id=ex["task_id"], prompt=ex["prompt"],
                     canonical_solution=ex["canonical_solution"],
                     test=ex["test"], entry_point=ex["entry_point"]) for ex in ds]
    rows = rows[:n] if n else rows
    return [{"task_id": r["task_id"], "model_prompt": r["prompt"],
             "code_prefix": r["prompt"], "test": r["test"],
             "entry_point": r["entry_point"]} for r in rows]


def _load_fix(n, offline):
    if offline:
        rows = list(SYNTHETIC_HUMANEVAL_FIX)
    else:
        from datasets import load_dataset
        ds = load_dataset("bigcode/humanevalpack", "python", split="test")
        rows = [dict(task_id=ex["task_id"], declaration=ex["declaration"],
                     buggy_solution=ex["buggy_solution"], test=ex["test"],
                     entry_point=ex["entry_point"]) for ex in ds]
    rows = rows[:n] if n else rows
    out = []
    for r in rows:
        decl = r.get("declaration", "")
        prompt = _FIX_TMPL.format(decl=decl, buggy=r["buggy_solution"])
        # model emits a full function -> nothing to prepend before running
        out.append({"task_id": r["task_id"], "model_prompt": prompt,
                    "code_prefix": "", "test": r["test"],
                    "entry_point": r["entry_point"]})
    return out