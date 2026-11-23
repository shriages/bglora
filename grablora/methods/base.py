from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..core import LayerRef, MethodSpec, Step

METHODS: dict[str, type["Method"]] = {}


def register(cls):
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a class-level `name`")
    METHODS[cls.name] = cls
    return cls


def build(name: str, spec: MethodSpec | None = None) -> "Method":
    if name not in METHODS:
        raise KeyError(f"unknown method '{name}'; have {sorted(METHODS)}")
    cls = METHODS[name]
    return cls(spec or MethodSpec(name=name, **cls.default_spec_kwargs()))


class Method(ABC):
    name: str = ""
    kind: str = "factored"          # "full" | "factored" | "proposal"
    needs_per_sample: bool = False
    needs_init_gradient: bool = False    # does init_layer() use G0?

    #: per-method spec defaults, merged into MethodSpec by build()
    @staticmethod
    def default_spec_kwargs() -> dict:
        return {}

    def __init__(self, spec: MethodSpec):
        self.spec = spec
        self.state: dict[str, object] = {}
        #: device the method's math runs on (set by the training loop). bg streams
        #: its CPU-stored per-sample evidence here per layer so the heavy SVD/eigh
        #: matmuls run on the GPU instead of the CPU.
        self.compute_device = "cpu"

    @property
    def refresh_interval(self) -> int:
        """Steps between full subspace re-estimations. 1 = every step (default). bg
        overrides via spec so the expensive per-sample evidence + Grassmann rotation
        run every N steps, with a cheap frozen-subspace update in between."""
        return int(getattr(self.spec, "refresh_interval", 1) or 1)

    # ---- setup -------------------------------------------------------------
    @abstractmethod
    def init_layer(self, ref: LayerRef, W0: torch.Tensor,
                   G0: torch.Tensor | None) -> torch.Tensor:
        """Store this layer's state; return the initial EFFECTIVE weight to
        install in the model. W0, G0 are float32 (d_out, d_in)."""

    # ---- one step ----------------------------------------------------------
    @abstractmethod
    def step(self, ref: LayerRef, G, is_refresh: bool = True) -> Step:
        """G is a Tensor (bg on a NON-refresh step) or a list of Tensors (bg
        per-sample evidence on a refresh step). `is_refresh` tells bg
        whether to re-estimate/rotate the subspace this step. Return a Step whose
        dW_lin is the raw induced direction and whose commit(lr) applies the step at
        learning rate lr and returns the exact dW."""

    # ---- reporting ---------------------------------------------------------
    def subspace(self, key: str):
        """(U, V) orthonormal bases of the current rank-r tangent space, or None.
        Used by the coverage diagnostic. bg_lora reports its tracked U, V."""
        return None