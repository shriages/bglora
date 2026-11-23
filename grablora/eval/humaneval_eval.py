from __future__ import annotations

import re
import multiprocessing as _mp

import torch

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, code_prefix: str = "") -> str:
    """Pull the model's code out of its (possibly chatty) output. Prefer a fenced
    ```python block; otherwise cut the generation at the first line that looks like
    prose after code / a new top-level def when a prefix was given."""
    m = _FENCE.search(text)
    if m:
        return m.group(1)
    # no fence: in completion mode the model continues the signature, so keep the raw
    # continuation up to an obvious stop; in fix mode we still return the raw text.
    stops = ["\nclass ", "\nprint(", "\nif __name__", "\n```", "\n#", "\nassistant"]
    cut = len(text)
    for s in stops:
        j = text.find(s)
        if j != -1:
            cut = min(cut, j)
    return text[:cut]


def _program(row, completion: str) -> str:
    prefix = row.get("code_prefix", "")
    body = completion if prefix == "" else (prefix + completion)
    return body + "\n\n" + row["test"] + f"\n\ncheck({row['entry_point']})\n"


def _isolate_streams():
    """Close stdin (input() -> EOFError, never blocks) and swallow stdout/stderr at the
    FD level (so even C-level writes / os.write are silenced), before running candidate
    code. The wall-clock timeout in check_correctness is the backstop for infinite loops."""
    import os
    import sys
    try:
        fd_in = os.open(os.devnull, os.O_RDONLY)
        os.dup2(fd_in, 0)                       # stdin from /dev/null -> input() hits EOF
    except Exception:
        pass
    try:
        fd_out = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd_out, 1)                      # swallow stdout
        os.dup2(fd_out, 2)                      # swallow stderr
    except Exception:
        pass
    try:
        sys.stdin = open(os.devnull, "r")       # belt-and-suspenders for input()
    except Exception:
        pass


def _worker(program: str, q):
    _isolate_streams()
    _reliability_guard()
    try:
        env = {"__name__": "__main__"}
        exec(compile(program, "<candidate>", "exec"), env)
        q.put(True)
    except BaseException:
        q.put(False)


def check_correctness(row, completion: str, timeout: float = 8.0) -> bool:
    program = _program(row, completion)
    # fork (Linux default) keeps the child cheap and avoids re-importing __main__;
    # spawn is the fallback where fork is unavailable.
    try:
        ctx = _mp.get_context("fork")
    except ValueError:
        ctx = _mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(program, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return False
    try:
        return bool(q.get_nowait())
    except Exception:
        return False


@torch.no_grad()
def evaluate_humaneval(model, tok, rows, max_new_tokens=512, batch_size=8,
                       device="cuda", timeout=8.0):
    """rows are executor-ready (see data/humaneval.py). Returns (pass@1, name, preds)."""
    was_cache, was_train = model.config.use_cache, model.training
    model.config.use_cache = True
    model.eval()
    tok.padding_side = "left"
    completions = []
    try:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            prompts = [r["model_prompt"] for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            cont = gen[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(cont, skip_special_tokens=True)
            completions.extend(texts)
            del enc, gen, cont
    finally:
        model.config.use_cache = was_cache
        model.train(was_train)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    passed, preds = 0, []
    for r, txt in zip(rows, completions):
        code = extract_code(txt, r.get("code_prefix", ""))
        ok = check_correctness(r, code, timeout=timeout)
        passed += int(ok)
        preds.append({"task_id": r["task_id"], "passed": bool(ok), "completion": code[:2000]})
    return passed / max(len(rows), 1), "pass@1", preds


def _reliability_guard():
    import os
    import builtins
    import faulthandler
    faulthandler.disable()
    builtins.exit = None
    builtins.quit = None
    builtins.input = None          # example-usage input() -> TypeError (caught), never blocks
    builtins.breakpoint = None     # a stray breakpoint() would otherwise hang on the debugger
    os.environ["OMP_NUM_THREADS"] = "1"
    for fn in ("kill", "system", "remove", "removedirs", "rmdir", "unlink"):
        if hasattr(os, fn):
            try:
                setattr(os, fn, None)
            except Exception:
                pass
    import shutil
    for fn in ("rmtree", "move"):
        if hasattr(shutil, fn):
            setattr(shutil, fn, None)
    import subprocess
    subprocess.Popen = None