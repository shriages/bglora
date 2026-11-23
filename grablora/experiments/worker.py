from __future__ import annotations

import json
import sys
import traceback


def main(argv):
    if len(argv) < 2:
        print("usage: python -m grablora.experiments.worker <job.json>", file=sys.stderr)
        return 2
    job_path = argv[1]
    with open(job_path) as f:
        job = json.load(f)

    # payload may still carry _-prefixed selection keys if the parent didn't strip them;
    # drop them defensively so RunConfig only sees real fields.
    payload = {k: v for k, v in job["payload"].items() if not k.startswith("_")}
    base_out = job["base_out"]
    result_path = job_path + ".result"

    try:
        import torch
        from grablora.experiments.run_task import run_task, RunConfig

        # TF32 OFF by default: bg_lora's filter feeds eigh/svd in float32, and TF32 would
        # perturb the subspace estimate and could reorder methods. Only enable if the
        # payload explicitly asks (allow_tf32=True).
        tf32 = bool(payload.get("allow_tf32", False))
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32

        cfg = RunConfig(**payload)
        summary = run_task(cfg, base_out=base_out)
        out = {"ok": True, "summary": summary}
    except BaseException as e:                       # report everything, never raise
        out = {"ok": False,
               "error": f"{type(e).__name__}: {e}"[:500],
               "traceback": traceback.format_exc()[-4000:]}
        traceback.print_exc()                        # also goes to the per-cell .log

    with open(result_path, "w") as f:
        json.dump(out, f)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))