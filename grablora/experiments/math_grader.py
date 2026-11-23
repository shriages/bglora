from __future__ import annotations

import re


# ---- boxed extraction (brace-balanced) ---------------------------------------
def last_boxed_only_string(string: str):
    """Return the last '\\boxed{...}' (or '\\fbox{...}') substring, braces balanced, or None."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    depth = 0
    right = None
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    return string[idx:right + 1] if right is not None else None


def remove_boxed(s: str):
    """Strip the '\\boxed{...}' wrapper, returning the inside."""
    if s is None:
        return None
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    m = re.match(r"\\(boxed|fbox)\s*{(.*)}$", s, flags=re.DOTALL)
    return m.group(2) if m else s


def extract_boxed_answer(text: str):
    """The inside of the last \\boxed{} in `text`, or None."""
    b = last_boxed_only_string(text)
    return remove_boxed(b) if b else None


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _last_number(text: str):
    m = _NUM_RE.findall(text.replace(",", ""))
    return m[-1] if m else None


# ---- normalization -----------------------------------------------------------
def _strip_string(s: str) -> str:
    s = s.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").lstrip(".") if s.startswith(".") else s
    s = s.replace("\\text{", "").replace("\\text", "").replace("\\mathrm", "")
    s = s.replace("\\,", "").replace(" ", "")
    s = s.replace("\\{", "").replace("\\}", "").replace("{", "").replace("}", "")
    # \frac a b / \fracab -> a/b
    s = re.sub(r"\\d?frac{([^}]*)}{([^}]*)}", r"\1/\2", s)
    s = re.sub(r"\\frac([0-9])([0-9])", r"\1/\2", s)
    s = re.sub(r"\\sqrt{([^}]*)}", r"sqrt(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.rstrip(".")
    if s.endswith("/1"):
        s = s[:-2]
    return s.strip()


def _num(x: str):
    try:
        if "/" in x:
            a, b = x.split("/", 1)
            return float(a) / float(b)
        return float(x)
    except (ValueError, ZeroDivisionError):
        return None


def is_equiv(a: str, b: str) -> bool:
    """MATH-style answer equivalence: normalized string match, with a numeric fallback."""
    if a is None or b is None:
        return False
    na, nb = _strip_string(a), _strip_string(b)
    if na == nb:
        return True
    va, vb = _num(na), _num(nb)
    if va is not None and vb is not None:
        return abs(va - vb) < 1e-6
    return False


def grade(pred_text: str, gold: str) -> bool:
    """True if the model's answer matches the gold answer.

    pred_text : the model's full generation (we take its last \\boxed{}, else last number).
    gold      : the reference solution containing \\boxed{}, or a bare answer string.
    """
    gold_ans = extract_boxed_answer(gold)
    if gold_ans is None:                       # gold is already a bare answer
        gold_ans = gold
    pred_ans = extract_boxed_answer(pred_text)
    if pred_ans is None:                       # model didn't box -> fall back to last number
        pred_ans = _last_number(pred_text)
    return is_equiv(pred_ans, gold_ans)