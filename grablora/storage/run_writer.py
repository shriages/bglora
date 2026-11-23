"""The structured run directory:

    runs/{task}/{model}/{method}/seed_{seed}/
        config.yaml            full experiment config (experiment/model/dataset/training/adapter/bg_lora)
        environment.json       python/torch/cuda/gpu/lib versions + git commit + timestamp
        dataset_manifest.json  dataset id/splits/prompt/counts + a content hash (reproducibility)
        train_metrics.jsonl    one line per step
        eval_metrics.jsonl     one line per checkpoint (val_ce, accuracy, eval_loss, ...)
        resource_metrics.jsonl one line per checkpoint (gpu mem, tokens/s, wall)
        geometry/checkpoint_{step:06d}.parquet   per-layer subspace geometry at each checkpoint
        z_evidence/refresh_{step:06d}.parquet    RAW per-atom coordinate samples z[i,k] at each
                                                 refresh -- bg_lora only -- for the offline
                                                 Gaussian-assumption (normality + mean) tests
        predictions/           generated predictions (final, and best) as jsonl
        checkpoints/           compact adapter state, ONLY if save_checkpoints (see note)
        summary.json           the headline numbers

DELIBERATE SHORTENINGS (you said we may). Inline evaluation already records
accuracy/val_ce/geometry at every checkpoint, so re-loading a saved model to
re-test it is redundant; `checkpoints/` is therefore OFF by default and, when on,
stores only the COMPACT adapter state (factors / U,V,m,P), not a full 0.5B copy per
step -- which at 345 runs x many steps would be enormous for no analysis gain. The
per-checkpoint research signal lives in `geometry/`.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time


def _slug(s: str) -> str:
    return str(s).replace("/", "--").replace(" ", "_")


def run_dir(base: str, task: str, model: str, method: str, seed, variant: str = "") -> str:
    """The directory a run WOULD write to -- for skip-if-exists checks, without
    creating anything. Matches RunWriter's layout exactly."""
    leaf = f"seed_{seed}" + (f"__{_slug(variant)}" if variant else "")
    return os.path.join(base, _slug(task), _slug(model), _slug(method), leaf)


class RunWriter:
    def __init__(self, base: str, task: str, model: str, method: str, seed: int,
                 variant: str = ""):
        # `variant` disambiguates sweep cells that share (task, model, method, seed)
        # -- e.g. bg_lora's kappa/q grid. Without it every cell collides on one dir.
        leaf = f"seed_{seed}" + (f"__{_slug(variant)}" if variant else "")
        self.dir = os.path.join(base, _slug(task), _slug(model), _slug(method), leaf)
        self.geom_dir = os.path.join(self.dir, "geometry")
        self.ckpt_dir = os.path.join(self.dir, "checkpoints")
        self.pred_dir = os.path.join(self.dir, "predictions")
        for d in (self.dir, self.geom_dir, self.pred_dir):
            os.makedirs(d, exist_ok=True)
        self._fh = {}

    # ---- streaming jsonl ----
    def _line(self, name: str, row: dict) -> None:
        fh = self._fh.get(name)
        if fh is None:
            fh = self._fh[name] = open(os.path.join(self.dir, name), "a")
        fh.write(json.dumps(row) + "\n")
        fh.flush()

    def train(self, row: dict):    self._line("train_metrics.jsonl", row)
    def eval(self, row: dict):     self._line("eval_metrics.jsonl", row)
    def resource(self, row: dict): self._line("resource_metrics.jsonl", row)

    # ---- one-shot files ----
    def config(self, cfg: dict) -> None:
        import yaml
        with open(os.path.join(self.dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def environment(self, extra: dict | None = None) -> None:
        import torch
        env = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        }
        for mod in ("transformers", "datasets", "peft", "accelerate"):
            try:
                env[mod] = __import__(mod).__version__
            except Exception:
                env[mod] = None
        if extra:
            env.update(extra)
        with open(os.path.join(self.dir, "environment.json"), "w") as f:
            json.dump(env, f, indent=2)

    def dataset_manifest(self, m: dict) -> None:
        with open(os.path.join(self.dir, "dataset_manifest.json"), "w") as f:
            json.dump(m, f, indent=2)

    def geometry(self, step: int, records: list) -> None:
        import pandas as pd
        pd.DataFrame(records).to_parquet(
            os.path.join(self.geom_dir, f"checkpoint_{step:06d}.parquet"), index=False)

    def z_evidence(self, records: list) -> None:
        """RAW bg_lora coordinate samples for the offline normality / mean tests.

        `records` is a FLAT list of per-(layer, atom, refresh) dicts produced by
        BGLoRA._buffer_z (each carries the n raw samples z[:,k] plus sigma0_k, zbar_k,
        m_k, beta_k, std_k, rho_k). They are grouped by `step` and written one parquet
        per refresh step -- z_evidence/refresh_{step:06d}.parquet -- mirroring the
        geometry/ layout. No-op on an empty list. Safe to call repeatedly: each refresh
        step is drained once, but if a step's rows ever arrive split across calls they
        are concatenated into the existing file rather than overwriting it."""
        if not records:
            return
        import pandas as pd
        zdir = os.path.join(self.dir, "z_evidence")
        os.makedirs(zdir, exist_ok=True)
        by_step: dict = {}
        for rec in records:
            by_step.setdefault(int(rec.get("step", -1)), []).append(rec)
        for step, recs in by_step.items():
            path = os.path.join(zdir, f"refresh_{step:06d}.parquet")
            df = pd.DataFrame(recs)
            if os.path.exists(path):
                try:
                    df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
                except Exception:
                    pass
            df.to_parquet(path, index=False)

    def predictions(self, tag: str, preds: list) -> None:
        with open(os.path.join(self.pred_dir, f"{tag}.jsonl"), "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

    def checkpoint(self, step: int, state: dict) -> None:
        import torch
        d = os.path.join(self.ckpt_dir, f"step_{step:06d}")
        os.makedirs(d, exist_ok=True)
        torch.save(state, os.path.join(d, "adapter_state.pt"))

    def summary(self, s: dict) -> None:
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(s, f, indent=2)

    def close(self):
        for fh in self._fh.values():
            fh.close()
        self._fh = {}


def _git_commit():
    for src in (lambda: os.environ.get("GRABLORA_GIT_COMMIT"),
                lambda: subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()):
        try:
            v = src()
            if v:
                return v
        except Exception:
            pass
    return None


def content_hash(items) -> str:
    h = hashlib.sha1()
    for x in items:
        h.update(str(x).encode())
    return h.hexdigest()[:16]