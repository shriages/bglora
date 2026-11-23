from __future__ import annotations

import re

import torch
import torch.nn as nn

from ..core import LayerRef

# ---------------------------------------------------------------- causal (Qwen/Llama)
_CANON = {
    "q_proj": "attn.q", "k_proj": "attn.k", "v_proj": "attn.v", "o_proj": "attn.o",
    "gate_proj": "mlp.gate", "up_proj": "mlp.up", "down_proj": "mlp.down",
}
PRESETS = {
    "attn_qv":   {"attn.q", "attn.v"},
    "attn_qkv":  {"attn.q", "attn.k", "attn.v"},
    "attn_all":  {"attn.q", "attn.k", "attn.v", "attn.o"},
    "all_linear": set(_CANON.values()),
}

# ---------------------------------------------------------------- seq2seq (T5 family)
# T5 module paths look like:
#   encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}
#   decoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}
#   decoder.block.{i}.layer.1.EncDecAttention.{q,k,v,o}      (cross-attention)
#   {encoder,decoder}.block.{i}.layer.{1,2}.DenseReluDense.{wi,wo}
# The leaf name (q/k/v/o) repeats across all three attention types, so we canonicalise
# on ATTENTION TYPE, not the bare leaf. Canonical names:
#   enc.attn.{q,k,v,o}   dec.attn.{q,k,v,o}   dec.cross.{q,k,v,o}
#   enc.mlp.{wi,wo}      dec.mlp.{wi,wo}
_T5_ATTN_LEAVES = ("q", "k", "v", "o")


def is_seq2seq(model_id: str) -> bool:
    """True for encoder-decoder models this harness supports via the T5 path.
    Matches the T5 family by id substring (t5, flan-t5, mt5, byt5, ul2). Cheap and
    dependency-free so callers can branch without loading the model."""
    s = (model_id or "").lower()
    return any(tok in s for tok in ("t5", "flan-t5", "flan_t5", "mt5", "byt5", "ul2"))


def _t5_canon(path: str):
    """Canonical name for a T5 Linear path, or None if it's not an adapt target.
    Keys on stack (encoder/decoder) + attention type so repeated q/k/v/o leaves in
    different stacks/attentions do not collide."""
    stack = "enc" if path.startswith("encoder.") else ("dec" if path.startswith("decoder.") else None)
    if stack is None:
        return None                                  # lm_head, shared embeddings, etc.
    leaf = path.split(".")[-1]
    if "SelfAttention" in path and leaf in _T5_ATTN_LEAVES:
        return f"{stack}.attn.{leaf}"
    if "EncDecAttention" in path and leaf in _T5_ATTN_LEAVES:
        return f"{stack}.cross.{leaf}"               # decoder-only in practice
    if "DenseReluDense" in path and leaf in ("wi", "wo"):
        return f"{stack}.mlp.{leaf}"
    return None


T5_PRESETS = {
    # Match the classic LoRA-on-T5 recipe: adapt attention q & v across BOTH stacks
    # (encoder self-attn + decoder self-attn). Cross-attention and MLP left frozen.
    "attn_qv":   {"enc.attn.q", "enc.attn.v", "dec.attn.q", "dec.attn.v"},
    "attn_qkv":  {"enc.attn.q", "enc.attn.k", "enc.attn.v",
                  "dec.attn.q", "dec.attn.k", "dec.attn.v"},
    # attention across self AND cross attention, both stacks
    "attn_all":  {f"{s}.attn.{l}" for s in ("enc", "dec") for l in _T5_ATTN_LEAVES}
                 | {f"dec.cross.{l}" for l in _T5_ATTN_LEAVES},
    "all_linear": ({f"{s}.attn.{l}" for s in ("enc", "dec") for l in _T5_ATTN_LEAVES}
                   | {f"dec.cross.{l}" for l in _T5_ATTN_LEAVES}
                   | {f"{s}.mlp.{l}" for s in ("enc", "dec") for l in ("wi", "wo")}),
}

_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _canon(path: str):
    return _CANON.get(path.split(".")[-1])


def block_of(path: str) -> int:
    """Transformer block index. For T5, encoder.block.3 and decoder.block.3 both
    contain '.3.'; the caller must namespace by stack, so we fold the stack into the
    returned index (decoder blocks offset by +1000) to keep enc/dec blocks distinct
    when n_blocks sub-selects. Causal models are unaffected (no encoder./decoder.)."""
    m = re.search(r"\.block\.(\d+)\.", path)         # T5 uses '.block.{i}.'
    if m is None:
        m = re.search(r"\.(\d+)\.", path)            # causal: '.layers.{i}.'
    if m is None:
        return -1
    idx = int(m.group(1))
    if path.startswith("decoder."):
        idx += 1000                                  # keep dec blocks distinct from enc
    return idx


def load_model_and_tokenizer(model_id: str, dtype: str = "bf16", device: str = "cuda",
                             trust_remote_code: bool = True, revision=None,
                             attn_implementation=None, gradient_checkpointing=True):
    from transformers import AutoTokenizer
    seq2seq = is_seq2seq(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code,
                                        revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Causal models generate by continuing the prompt -> LEFT pad so the last real
    # token is at a fixed position. T5's encoder is not autoregressive; standard T5
    # tokenisation is RIGHT-padded, and the eval path sets padding_side explicitly.
    tok.padding_side = "left" if not seq2seq else "right"
    kw = {}
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    if seq2seq:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=_DTYPE[dtype], trust_remote_code=trust_remote_code,
            revision=revision, **kw)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=_DTYPE[dtype], trust_remote_code=trust_remote_code,
            revision=revision, **kw)
    model.to(device)
    model.config.use_cache = False
    if gradient_checkpointing:
        # freeze_and_arm() leaves ONLY q/v proj trainable, so the base (embeddings ->
        # first block) requires no grad. HF's DEFAULT (reentrant) checkpointing needs a
        # grad-requiring input to trigger, so on a frozen base it silently no-ops and
        # saves NO memory. use_reentrant=False checkpoints regardless; on older
        # transformers that don't accept the kwarg, enable_input_require_grads() makes
        # the reentrant variant engage instead. Either way checkpointing actually fires.
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
    return model, tok


def find_targets(model: nn.Module, preset: str = "attn_qv", n_blocks: int = 0):
    """Return (mods, refs): dict key->nn.Linear and key->LayerRef for the preset.
    n_blocks>0 keeps only that many evenly-spaced transformer blocks (cheaper).

    Dispatches on architecture: T5 (encoder-decoder) uses the T5 name map + presets;
    everything else uses the causal Qwen/Llama map. Detection is by module structure
    (presence of `.encoder.block.` AND `.decoder.block.`), so it does not depend on
    the model id string here."""
    module_paths = [p for p, _ in model.named_modules()]
    is_t5 = any(".encoder.block." in p or p.startswith("encoder.block.")
                for p in module_paths) and \
            any(".decoder.block." in p or p.startswith("decoder.block.")
                for p in module_paths)

    if is_t5:
        wanted = T5_PRESETS[preset]
        canon = _t5_canon
    else:
        wanted = PRESETS[preset]
        canon = lambda p: _CANON.get(p.split(".")[-1])

    mods, refs = {}, {}
    for path, m in model.named_modules():
        c = canon(path)
        if c is None or c not in wanted or not isinstance(m, nn.Linear):
            continue
        d_out, d_in = m.weight.shape
        mods[path] = m
        refs[path] = LayerRef(path, d_out, d_in)
    if not mods:
        present = sorted({c for p, _ in model.named_modules() if (c := canon(p))})
        raise ValueError(f"preset '{preset}' matched no nn.Linear. Present canon "
                         f"names: {present}")
    if n_blocks > 0:
        blocks = sorted({block_of(k) for k in mods})
        if n_blocks < len(blocks):
            keep = ({blocks[round(i * (len(blocks) - 1) / (n_blocks - 1))]
                     for i in range(n_blocks)} if n_blocks > 1 else {blocks[-1]})
            mods = {k: v for k, v in mods.items() if block_of(k) in keep}
            refs = {k: v for k, v in refs.items() if block_of(k) in keep}
    return mods, refs


def freeze_and_arm(model: nn.Module, target_mods: dict):
    """Freeze everything, then require grad on the target weights only, so one
    backward yields dL/dW for exactly those matrices (== the method's evidence)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in target_mods.values():
        m.weight.requires_grad_(True)