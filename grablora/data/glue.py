from __future__ import annotations

_G = "nyu-mll/glue"              

GLUE = {
    "sst2": dict(hf=(_G, "sst2"), fields=("sentence",),
                 template="Review: {sentence}\nIs the sentiment positive? Answer yes or no:",
                 yes_if=lambda l: l == 1, metric="accuracy"),
    "mrpc": dict(hf=(_G, "mrpc"), fields=("sentence1", "sentence2"),
                 template="Sentence 1: {sentence1}\nSentence 2: {sentence2}\n"
                          "Do they mean the same thing? Answer yes or no:",
                 yes_if=lambda l: l == 1, metric="accuracy"),
    "cola": dict(hf=(_G, "cola"), fields=("sentence",),
                 template="Sentence: {sentence}\nIs this sentence grammatical? Answer yes or no:",
                 yes_if=lambda l: l == 1, metric="accuracy"),
    "qnli": dict(hf=(_G, "qnli"), fields=("question", "sentence"),
                 template="Question: {question}\nSentence: {sentence}\n"
                          "Does the sentence answer the question? Answer yes or no:",
                 yes_if=lambda l: l == 0, metric="accuracy"),   # 0 = entailment
    "rte": dict(hf=(_G, "rte"), fields=("sentence1", "sentence2"),
                template="Premise: {sentence1}\nHypothesis: {sentence2}\n"
                         "Does the premise entail the hypothesis? Answer yes or no:",
                yes_if=lambda l: l == 0, metric="accuracy"),
    "qqp": dict(hf=(_G, "qqp"), fields=("question1", "question2"),
                template="Question 1: {question1}\nQuestion 2: {question2}\n"
                         "Are these two questions duplicates? Answer yes or no:",
                yes_if=lambda l: l == 1, metric="accuracy"),
}

VERBALIZERS = ["no", "yes"]        # index by yes(1)/no(0)

SYNTHETIC_GLUE = [
    {"label": 1, "sentence": "a wonderful, moving film", "sentence1": "a", "sentence2": "a",
     "question": "q", "question1": "q", "question2": "q"},
    {"label": 0, "sentence": "a dull and lifeless bore", "sentence1": "a", "sentence2": "b",
     "question": "q", "question1": "q", "question2": "r"},
] * 8


def load_glue(subset: str, split: str, n: int = 0, offline: bool = False):
    spec = GLUE[subset]
    if offline:
        rows = [{**r, "label01": int(spec["yes_if"](r["label"]))} for r in SYNTHETIC_GLUE]
        return rows[:n] if n else rows
    from datasets import load_dataset
    hf_split = {"train": "train", "val": "train", "test": "validation"}.get(split, split)
    ds = load_dataset(*spec["hf"], split=hf_split)
    rows = []
    for ex in ds:
        rows.append({**{f: ex[f] for f in spec["fields"]},
                     "label01": int(spec["yes_if"](ex["label"]))})
        if n and len(rows) >= n:
            break
    return rows


def format_glue_prompt(subset: str, ex: dict) -> str:
    spec = GLUE[subset]
    return spec["template"].format(**{f: ex[f] for f in spec["fields"]})


def metric_name(subset: str) -> str:
    return GLUE[subset]["metric"]