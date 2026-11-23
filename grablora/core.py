from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class LayerRef:
    """One weight matrix we adapt, in MATH orientation (d_out, d_in)."""
    key: str
    d_out: int
    d_in: int


@dataclass(frozen=True)
class MethodSpec:
    name: str = "base"

    # ---- adapter knobs ----
    r: int = 4                       # LoRA rank
    lora_alpha: float = 16.0
    use_rslora: bool = True

    # ---- bg_lora (Bayesian Grassmannian filter) ----
    kappa_u: float = 1e-3
    kappa_v: float = 1e-3
    q0: float = 10.0
    q1: float = 10.0
    lambda_shrink: float = 1.0
    lambda_par: float = 1.0
    lambda_perp: float = 1.0
    beta_pool: bool = False
    rediag: bool = True              # re-diagonalise the core each step
    refresh_interval: int = 20       # rotate the subspace every N steps (cheap frozen
                                     # filter in between) -- big speedup, little quality loss
    j_max: int = 1                   # alternating grassmann sweeps (warm-started; 1 is enough)
    tol: float = 1e-3
    evidence_n: int = 16             # per-sample gradient count (>=2; beta needs spread)
    evidence_bs: int = 8             # examples averaged inside one evidence sample

    # ---- optimizer (decoupled AdamW; the real learning rate is applied by the loop) ----
    adam_lr: float = 2e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.0

    # ---- bg_lora coefficient estimator ----
    #   "kalman" -- the Bayesian posterior mean (the method as published).
    #   "adam"   -- Adam-in-subspace with a MEASURED denominator (observation noise
    #               1/(n*beta) + rotation uncertainty P_pred); used for the
    #               preconditioner analysis figures.
    coeff_optimizer: str = "kalman"
    adam_step_cap: float = 1.0       # stability for coeff_optimizer="adam": |m| <= 1/cap

    def with_(self, **overrides) -> "MethodSpec":
        return replace(self, **overrides)


@dataclass
class Step:
    dW_lin: "object"                 # torch.Tensor
    commit: "object"                 # Callable[[float], torch.Tensor]
    info: dict = field(default_factory=dict)