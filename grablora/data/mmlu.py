from __future__ import annotations

LETTERS = ["A", "B", "C", "D"]

_HEADER = "The following are multiple choice questions (with answers) about {subject}.\n\n"


def _q_block(question: str, choices, answer_letter: str = None) -> str:
    lines = [question.strip()]
    for L, c in zip(LETTERS, choices):
        lines.append(f"{L}. {str(c).strip()}")
    lines.append("Answer:" + (f" {answer_letter}" if answer_letter else ""))
    return "\n".join(lines)


def _subject_name(subject: str) -> str:
    return (subject or "").replace("_", " ").strip() or "general knowledge"


SYNTHETIC_MMLU_DEV = [
    {"subject": "toy", "question": "2+2=?", "choices": ["3", "4", "5", "6"], "answer": 1},
    {"subject": "toy", "question": "Sky color?", "choices": ["Green", "Blue", "Red", "Pink"], "answer": 1},
]
SYNTHETIC_MMLU_TEST = [
    {"subject": "toy", "question": "1+1=?", "choices": ["1", "2", "3", "4"], "answer": 1},
    {"subject": "toy", "question": "Grass color?", "choices": ["Blue", "Green", "Red", "Grey"], "answer": 1},
] * 4


def load_mmlu_split(split: str, n: int = 0, source: str = "cais/mmlu",
                    offline: bool = False):
    if offline:
        rows = list(SYNTHETIC_MMLU_DEV if split == "dev" else SYNTHETIC_MMLU_TEST)
        return rows[:n] if n else rows
    from datasets import load_dataset
    ds = load_dataset(source, "all", split=split)
    rows = []
    for ex in ds:
        rows.append({"subject": ex.get("subject", ""),
                     "question": ex["question"],
                     "choices": list(ex["choices"]),
                     "answer": int(ex["answer"])})
        if n and len(rows) >= n:
            break
    return rows


def _dev_by_subject(dev_rows):
    d = {}
    for r in dev_rows:
        d.setdefault(r["subject"], []).append(r)
    return d


def format_mmlu_prompt(row: dict, n_shots: int, dev_by_subject: dict) -> str:
    subj = row["subject"]
    prompt = _HEADER.format(subject=_subject_name(subj))
    if n_shots:
        shots = (dev_by_subject.get(subj) or [])[:n_shots]
        for s in shots:
            prompt += _q_block(s["question"], s["choices"], LETTERS[s["answer"]]) + "\n\n"
    prompt += _q_block(row["question"], row["choices"], None)
    return prompt


def build_fewshot_index(dev_rows):
    return _dev_by_subject(dev_rows)