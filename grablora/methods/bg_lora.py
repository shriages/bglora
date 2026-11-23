"""Bayesian Grassmannian LoRA (GRAB-LoRA) -- the method itself.

GRAB-LoRA's identity is the whole trajectory: it re-reads the gradient every step,
lets the rank-r subspace ROTATE under a Grassmannian prior (manifold.grassmann_map),
and runs a per-atom Kalman filter over the coefficients. It proposes dW directly; the
training loop applies a real learning rate (dW = lr * dw).

COEFFICIENT ESTIMATOR (spec.coeff_optimizer):
  "adam"             -- Adam-in-subspace with a MEASURED denominator: observation noise
                        1/(n*beta) plus the rotation-aware uncertainty P_pred (rho enters
                        ONLY through P_pred). This variant produces the preconditioner
                        analysis (per-atom gain vs a plain-Adam gain, logged when
                        GRABLORA_PRECOND_TRACE=1).

WHY PER-SAMPLE EVIDENCE. beta is estimated from the spread of z_{i,k}=u_k^T Y_i v_k
across samples; an averaged gradient has no spread. This is the one place the harness
pays for unaveraged gradients (needs_per_sample = True), and why evidence_n is small.
"""
from __future__ import annotations

import torch

from ..core import LayerRef, Step
from . import manifold
from .base import Method, register


class _BGState:
    __slots__ = ("U", "V", "m", "P", "rho", "sigma0", "D", "beta", "mu", "v")

    def __init__(self, U, V, m, P, rho, sigma0, beta):
        self.U, self.V, self.m, self.P = U, V, m, P
        self.rho, self.sigma0 = rho, sigma0
        self.beta = beta                                 # last observation precision (per atom)
        self.D = torch.zeros(U.shape[0], V.shape[0], device=U.device)   # running delta
        # coeff_optimizer="adam": per-atom Adam moments over the coordinate.
        self.mu = torch.zeros_like(m)                    # 1st moment (momentum on zbar)
        self.v = torch.zeros_like(m)                     # 2nd moment (plain-Adam comparison)


@register
class BGLoRA(Method):
    name = "bg_lora"
    kind = "proposal"
    needs_per_sample = True

    def __init__(self, spec):
        super().__init__(spec)
        if spec.evidence_n < 2:
            raise ValueError("evidence_n must be >= 2: beta needs the sample spread of z.")
        # RAW EVIDENCE BUFFER. Filled at every refresh (and t=0) with the full per-sample
        # coordinate samples z[i,k]=u_k^T Y_i v_k, one row per (layer, atom, refresh), so
        # the Gaussian-observation assumption can be tested OFFLINE (normality of z[:,k],
        # and whether E[z_k] sits at sigma0_k). Drained by the run writer at each
        # checkpoint via pop_z_evidence().
        self._z_buffer: list = []
        self.cur_step = -1                    # set by the training loop; labels each record
        # PER-ATOM PRECONDITIONER TRACE (opt-in, coeff_optimizer="adam" only). At every
        # refresh (and t=0) record, per atom, the coefficient m_k, the denominator omega_k
        # and its parts (obs_var_k, P_pred_k), the posterior P_k, the alignment rho_k and
        # the right-subspace principal angle theta_V_k, so the per-DIRECTION preconditioner
        # story can be plotted offline. Enable with env GRABLORA_PRECOND_TRACE=1 (or
        # spec.precond_trace=True). Drained via pop_precond_trace() -> precond_trace.jsonl.
        import os as _os
        self._trace_on = bool(_os.environ.get("GRABLORA_PRECOND_TRACE")) or \
                         bool(getattr(spec, "precond_trace", False))
        self._precond_buffer: list = []
        self._last_precond = None

    def init_layer(self, ref, W0, G0):
        self.state[ref.key] = None            # built on the first step from per-sample evidence
        return W0.clone()

    # ---------------------------------------------------------------- coefficient
    def _coeff_update(self, st, zbar, beta, n, P_pred, m_pred, tstep):
        """Return the per-atom coefficient m; also set st.P (posterior variance) and,
        in adam mode, st.mu / st.v.

          coeff_optimizer="kalman" -> Bayesian posterior MEAN.
          coeff_optimizer="adam"   -> Adam 1st moment on the descent coordinate zbar,
                                      divided by a MEASURED denominator
                                      ( obs noise 1/(n*beta)  +  rotation unc. P_pred ),
                                      capped for stability. rho enters ONLY via P_pred.
        """
        spec = self.spec
        tstep = max(int(tstep), 1)
        if spec.coeff_optimizer == "adam":
            b1 = spec.adam_beta1
            b2 = spec.adam_beta2
            st.mu = b1 * st.mu + (1.0 - b1) * zbar               # first moment
            mu_hat = st.mu / (1.0 - b1 ** tstep)
            obs_var = 1.0 / (n * beta)                           # measured noise of the mean
            # Plain-Adam EMA second moment: a spectator here (never used in our step),
            # kept only to log the plain-Adam comparison curve for the SAME atom at the
            # SAME rotation angle in the preconditioner trace.
            st.v = b2 * st.v + (1.0 - b2) * zbar * zbar
            v_hat = st.v / (1.0 - b2 ** tstep)
            denom2 = obs_var + P_pred                            # measured denominator
            omega = denom2.clamp_min(0.0).sqrt()                 # replaces Adam's sqrt(v)
            m = mu_hat / (omega + spec.adam_eps)
            if spec.adam_step_cap > 0:                           # stability: |m| <= 1/cap
                lim = 1.0 / spec.adam_step_cap
                m = m.clamp(-lim, lim)
            st.P = 1.0 / (1.0 / P_pred + n * beta)               # posterior variance correction
            if self._trace_on:
                import torch as _t
                eps = float(spec.adam_eps)
                def _vec(x):
                    return x.detach().float().cpu() if _t.is_tensor(x) else _t.full_like(m, float(x)).detach().float().cpu()
                omega_adam = v_hat.clamp_min(0.0).sqrt()         # plain-Adam denom, same atoms
                om, oma = _vec(omega), _vec(omega_adam)
                self._last_precond = {"m": m.detach().float().cpu(), "omega": om,
                                      "omega_adam": oma,
                                      "gain": 1.0 / (om + eps),          # OUR 1/(sqrt(var)+eps)
                                      "gain_adam": 1.0 / (oma + eps),    # plain Adam's gain
                                      "obs_var": _vec(obs_var), "P_pred": _vec(P_pred),
                                      "P_post": _vec(st.P), "mu_hat": _vec(mu_hat)}
            return m
        # ---- kalman (posterior mean) ----
        obs_var = 1.0 / (n * beta)
        m = (obs_var / (P_pred + obs_var) * m_pred
             + P_pred / (P_pred + obs_var) * zbar)
        st.P = 1.0 / (1.0 / P_pred + n * beta)
        return m

    # ---------------------------------------------------------------- step
    def step(self, ref: LayerRef, Gs, is_refresh: bool = True) -> Step:
        spec = self.spec
        dev = self.compute_device

        # stream this layer's evidence to the compute device (GPU): the filter's
        # dense SVD/eigh matmuls are ~50-100x faster there than on the CPU.
        if isinstance(Gs, (list, tuple)):
            Gs = [g.to(dev, non_blocking=True) for g in Gs]
        else:
            Gs = Gs.to(dev, non_blocking=True)

        # ---- GUARD: sanitise non-finite evidence BEFORE any GPU eigh/svd ----
        # A NaN/Inf here (bf16 overflow on a hot step) would make cuSOLVER fault with an
        # async CUDA XID that the try/except in manifold.py cannot catch -- it kills the
        # context. nan_to_num keeps the run alive; the log names where it happened so you
        # can lower the LR if it fires often. lm_loop's finite-check on dW will still stop
        # the run cleanly if it is genuinely diverging.
        if isinstance(Gs, (list, tuple)):
            bad = [i for i, g in enumerate(Gs) if not torch.isfinite(g).all()]
            if bad:
                print(f"[bg_lora] non-finite evidence step={self.cur_step} "
                      f"layer={ref.key} samples={bad} -> nan_to_num")
                Gs = [torch.nan_to_num(g) for g in Gs]
        elif not torch.isfinite(Gs).all():
            print(f"[bg_lora] non-finite grad step={self.cur_step} layer={ref.key} -> nan_to_num")
            Gs = torch.nan_to_num(Gs)

        st: _BGState | None = self.state.get(ref.key)

        # z-statistics recorded at every refresh (None on frozen steps)
        z_stats: dict = {}

        if st is None:                         # ---- Algorithm 1, t = 0 (always a refresh)
            GsF = [(-g).float() for g in Gs]
            n = len(GsF)
            Ys = GsF
            Ybar = torch.stack(Ys).mean(0)
            U, S, Vh = torch.linalg.svd(Ybar, full_matrices=False)
            r = min(spec.r, S.numel())
            U, V, sigma = U[:, :r].contiguous(), Vh[:r].T.contiguous(), S[:r].clone()
            z = _coords(Ys, U, V)
            zbar = z.mean(0)
            beta = _beta(z, zbar, spec.beta_pool)
            if spec.coeff_optimizer == "adam":
                # prior uncertainty = one observation's variance; momentum starts 0.
                P0 = 1.0 / (n * beta)
                st = _BGState(U, V, torch.zeros_like(zbar), P0,
                              torch.ones(r), sigma, beta)
                self.state[ref.key] = st
                m = self._coeff_update(st, zbar, beta, n, P0, st.m, self.cur_step + 1)
                st.m = m
            else:
                alpha = spec.lambda_shrink * n * beta
                m = (alpha * sigma + n * beta * zbar) / (alpha + n * beta)
                P = 1.0 / (alpha + n * beta)
                st = _BGState(U, V, m, P, torch.ones(r), sigma, beta)
                self.state[ref.key] = st
            rot, iters = 0.0, 0
            # At t=0 there is no previous subspace, so rotation is undefined; use nan
            rot_rms_U = rot_rms_V = float("nan")
            theta_U = theta_V = torch.full((r,), float("nan"))
            z_stats = _z_stats(z, zbar, m)
            self._buffer_z(ref, z, zbar, m, beta, sigma, torch.ones(r, device=z.device), "init")
            self._buffer_precond(ref, st, theta_V, "init")
        elif is_refresh:                       # ---- Algorithm 2: rotate the subspace
            GsF = [(-g).float() for g in Gs]   # Gs is a LIST of per-sample gradients
            n = len(GsF)
            r = st.m.numel()
            Ys = [_reparam(G, st.U, st.V, spec.lambda_par, spec.lambda_perp) for G in GsF]
            Ybar = torch.stack(Ys).mean(0)
            U_raw, V_raw, iters = manifold.grassmann_map(
                Ys, st.U, st.V, spec.kappa_u, spec.kappa_v, r, spec.j_max, spec.tol)
            rot = manifold.projector_distance(U_raw, st.U)
            if spec.rediag:
                K = U_raw.T @ Ybar @ V_raw
                A_, _S, Bh_ = torch.linalg.svd(K)
                U_al = (U_raw @ A_).contiguous()
                V_al = (V_raw @ Bh_.T).contiguous()
            else:
                U_al, V_al = U_raw, V_raw
            M = manifold.atom_similarity(st.U, U_al, st.V, V_al)
            pi = manifold.assign(M)
            U_new, V_new = U_al[:, pi].contiguous(), V_al[:, pi].contiguous()
            signed = (st.U * U_new).sum(0) * (st.V * V_new).sum(0)
            flip = torch.where(signed < 0, -1.0, 1.0)
            U_new = (U_new * flip.unsqueeze(0)).contiguous()
            rho = signed.abs().clamp(0, 1)
            # ---- SUBSPACE-level principal angles (SORTED, NOT per-atom) ----
            sv_U = torch.linalg.svdvals((st.U.float().T @ U_new.float())).clamp(0.0, 1.0 - 1e-7)
            sv_V = torch.linalg.svdvals((st.V.float().T @ V_new.float())).clamp(0.0, 1.0 - 1e-7)
            theta_U = torch.acos(sv_U)              # principal angles U subspace (radians), sorted
            theta_V = torch.acos(sv_V)              # principal angles V subspace (radians), sorted
            rot_rms_U = float(theta_U.pow(2).mean().sqrt())
            rot_rms_V = float(theta_V.pow(2).mean().sqrt())
            # ---- PER-ATOM rotation angles (aligned to atom index k) ----
            cosU_atom = (st.U * U_new).sum(0).abs().clamp(0.0, 1.0 - 1e-7)
            cosV_atom = (st.V * V_new).sum(0).abs().clamp(0.0, 1.0 - 1e-7)
            theta_U_atom = torch.acos(cosU_atom)    # per-atom left  (U) rotation angle
            theta_V_atom = torch.acos(cosV_atom)    # per-atom right (V) rotation angle
            z = _coords(Ys, U_new, V_new)
            zbar = z.mean(0)
            beta = _beta(z, zbar, spec.beta_pool)
            q = spec.q0 + spec.q1 * (1.0 - rho ** 2)
            m_pred = rho * st.m
            P_pred = (rho ** 2) * st.P + q
            # coefficient (kalman posterior mean, or Adam-in-subspace). st.mu / st.P /
            # st.beta are indexed by atom position, and U_new/V_new are already reordered
            # (pi) and sign-aligned (flip) to that indexing, so momentum carries across the
            # rotation with no extra permute/flip -- the same convention m_pred = rho*st.m uses.
            m = self._coeff_update(st, zbar, beta, n, P_pred, m_pred, self.cur_step + 1)
            st.U, st.V, st.m, st.rho, st.beta = U_new, V_new, m, rho, beta
            z_stats = _z_stats(z, zbar, m)
            self._buffer_z(ref, z, zbar, m, beta, st.sigma0, rho, "refresh")
            self._buffer_precond(ref, st, theta_V_atom, "refresh", theta_U=theta_U_atom)
        else:                                  # ---- FROZEN step: subspace fixed, cheap
            # Gs is a SINGLE full-batch gradient. No rotation, no per-sample evidence.
            G = (-Gs).float() if not isinstance(Gs, (list, tuple)) else (-Gs[0]).float()
            Ybar = _reparam(G, st.U, st.V, spec.lambda_par, spec.lambda_perp)
            zbar = _coords([Ybar], st.U, st.V)[0]        # (r,)
            beta = st.beta
            rot, iters = 0.0, 0
            rot_rms_U = rot_rms_V = 0.0
            theta_U = theta_V = torch.zeros(st.m.numel())
            # Standard frozen step: subspace fixed (rho = 1), n = 1 observation.
            P_pred = st.P + spec.q0
            st.m = self._coeff_update(st, zbar, beta, 1, P_pred, st.m, self.cur_step + 1)
            st.rho = torch.ones_like(st.rho)

        # (g) proposal: filtered core + the two tangent strips
        U_, V_ = st.U, st.V
        core = (U_ * st.m.unsqueeze(0)) @ V_.T
        UtY = U_.T @ Ybar
        YV = Ybar @ V_
        UtYV = U_.T @ YV
        strips = U_ @ (UtY - UtYV @ V_.T) + (YV - U_ @ UtYV) @ V_.T
        dw = core + strips
        info = {"rho": float(st.rho.mean()), "m_abs": float(st.m.abs().mean()),
                "rotation": float(rot), "map_iters": int(iters),
                "rotation_rms_U": rot_rms_U,
                "rotation_rms_V": rot_rms_V,
                "theta_U": [round(float(x), 6) for x in theta_U.tolist()],
                "theta_V": [round(float(x), 6) for x in theta_V.tolist()],
                **z_stats}

        return Step(dW_lin=dw, commit=lambda lr: lr * dw, info=info)

    def subspace(self, key: str):
        st: _BGState | None = self.state.get(key)
        return None if st is None else (st.U, st.V)

    # ------------------------------------------------ raw z-evidence (offline tests)
    def _buffer_z(self, ref, z, zbar, m, beta, sigma0, rho, branch: str) -> None:
        """Record the RAW per-sample coordinates for this layer/refresh so the Gaussian
        assumption can be tested later without re-running anything.

        z : (n, r) tensor, z[i,k] = u_k^T Y_i v_k across the n per-sample evidence grads.
        One row is emitted per atom k, carrying the n samples plus the quantities the two
        tests need (normality of the n samples; mean check zbar_k vs sigma0_k)."""
        if z.shape[0] < 2:                     # need spread; frozen/singleton steps have none
            return
        z_cpu = z.detach().float().cpu()
        n, r = int(z_cpu.shape[0]), int(z_cpu.shape[1])
        zbar_l = zbar.detach().float().cpu().tolist()
        m_l    = m.detach().float().cpu().tolist()
        beta_l = beta.detach().float().cpu().tolist()
        sig_l  = sigma0.detach().float().cpu().tolist()
        rho_l  = rho.detach().float().cpu().tolist()
        step   = int(getattr(self, "cur_step", -1))
        for k in range(r):
            bk = float(beta_l[k])
            self._z_buffer.append({
                "step":     step,
                "layer":    str(ref.key),
                "branch":   branch,            # "init" (t=0) or "refresh"
                "atom":     k,                 # direction index
                "n":        n,                 # sample size (== evidence_n)
                "sigma0_k": float(sig_l[k]),   # fixed init singular value (mean reference)
                "zbar_k":   float(zbar_l[k]),  # sample mean of z[:,k]
                "m_k":      float(m_l[k]),     # posterior mean
                "beta_k":   bk,                # observation precision (1/var)
                "std_k":    bk ** -0.5 if bk > 0 else float("nan"),  # residual std = 1/sqrt(beta)
                "rho_k":    float(rho_l[k]),   # atom alignment |cos| to the previous subspace
                "z":        [float(v) for v in z_cpu[:, k].tolist()],  # the n raw samples
            })

    def pop_z_evidence(self) -> list:
        """Return and clear the buffered raw z records. The run writer calls this at each
        checkpoint (and once more at the end) and flushes them to z_evidence/*.parquet."""
        buf = self._z_buffer
        self._z_buffer = []
        return buf

    def _buffer_precond(self, ref, st, theta_V, phase: str, theta_U=None) -> None:
        """Emit one row per atom for the preconditioner trace, from the values stashed by
        the last _coeff_update (adam mode only). No-op unless tracing is on and a stash
        exists (kalman mode never sets it).

        theta_V / theta_U here are the PER-ATOM rotation angles (arccos|<prev_k,new_k>|)."""
        import torch as _t
        lp = self._last_precond
        if not self._trace_on or lp is None:
            return
        step = int(getattr(self, "cur_step", -1))
        rho = st.rho.detach().float().cpu() if _t.is_tensor(st.rho) else None
        th = theta_V.detach().float().cpu() if _t.is_tensor(theta_V) else None
        thu = theta_U.detach().float().cpu() if _t.is_tensor(theta_U) else None
        # process noise actually used this step: q_{t,k} = q0 + q1*(1 - rho_k^2).
        q0 = float(getattr(self.spec, "q0", 0.0))
        q1 = float(getattr(self.spec, "q1", 0.0))
        m = lp["m"]
        for k in range(m.numel()):
            rho_k = float(rho[k]) if rho is not None and k < rho.numel() else float("nan")
            q_k = q0 + q1 * (1.0 - rho_k ** 2) if rho_k == rho_k else q0   # NaN-safe
            self._precond_buffer.append({
                "step": step, "layer": ref.key, "phase": phase, "atom": int(k),
                "m": float(m[k]), "abs_m": float(m[k].abs()),
                "omega": float(lp["omega"][k]),
                "omega_adam": float(lp["omega_adam"][k]),
                "gain": float(lp["gain"][k]),               # OUR  1/(sqrt(var)+eps) = step multiplier
                "gain_adam": float(lp["gain_adam"][k]),     # plain-Adam gain, same angle
                "q": q_k,                                   # process noise q_{t,k} = q0 + q1*(1-rho^2)
                "obs_var": float(lp["obs_var"][k]),         # R_{t,k}: evidence noise 1/(n*beta)
                "P_pred": float(lp["P_pred"][k]),           # P_{t,k}^-: predictive variance (rho^2 P + q)
                "P_post": float(lp["P_post"][k]),           # P_{t,k}: posterior variance
                "mu_hat": float(lp["mu_hat"][k]),
                "rho": rho_k,
                "theta_V": (float(th[k]) if th is not None and k < th.numel() else float("nan")),
                "theta_U": (float(thu[k]) if thu is not None and k < thu.numel() else float("nan")),
            })
        self._last_precond = None                 # consumed

    def pop_precond_trace(self) -> list:
        """Return and clear the per-atom preconditioner trace. Run writer flushes it to
        precond_trace.jsonl. Empty unless GRABLORA_PRECOND_TRACE=1 (or spec.precond_trace)."""
        buf = self._precond_buffer
        self._precond_buffer = []
        return buf


# ------------------------------------------------------------------ helpers
def _reparam(G, U, V, lam_par, lam_perp):
    if lam_par == lam_perp:
        return G if lam_par == 1.0 else lam_par * G
    UtG = U.T @ G
    GV = G @ V
    UtGV = U.T @ GV
    G_perp = G - U @ UtG - GV @ V.T + U @ UtGV @ V.T
    return lam_par * G + (lam_perp - lam_par) * G_perp


def _coords(Ys, U, V):
    """z[i,k] = u_k^T Y_i v_k = diag(U^T Y_i V)."""
    return torch.stack([torch.einsum("ij,jk,ki->i", U.T, Y, V) for Y in Ys])


def _z_stats(z: torch.Tensor, zbar: torch.Tensor, m: torch.Tensor) -> dict:
    """Scalar summaries of the per-sample coordinates z[i,k] for later statistical
    testing (normality + mean). Surfaced into train_rows via step info."""
    n = z.shape[0]
    if n < 2:
        return {}
    z_f = z.float()
    zbar_f = zbar.float()
    std = (z_f - zbar_f).pow(2).mean(0).sqrt()          # per-atom std across samples
    skew = ((z_f - zbar_f).pow(3).mean(0)
            / std.clamp_min(1e-30).pow(3))               # per-atom skewness
    return {
        "z_mean_abs":  float(zbar_f.abs().mean()),
        "z_std":       float(std.mean()),
        "z_resid_abs": float((zbar_f - m.float()).abs().mean()),
        "z_skew":      float(skew.mean()),
        "z_n":         n,                                # number of samples (evidence_n)
    }


def _beta(z, zbar, pool: bool):
    """1/beta_k = unbiased sample variance of z across samples, floored. Pool over
    atoms if requested."""
    n = z.shape[0]
    if pool:
        var = (z - zbar).pow(2).sum() / (z.numel() - 1)
        scale = z.pow(2).mean().clamp_min(1e-30)
        return (1.0 / var.clamp_min(1e-12 * scale).clamp_min(1e-30)).expand(z.shape[1]).clone()
    var = (z - zbar).pow(2).sum(0) / (n - 1)
    scale = z.pow(2).mean(0).clamp_min(1e-30)
    return 1.0 / var.clamp_min(1e-12 * scale).clamp_min(1e-30)