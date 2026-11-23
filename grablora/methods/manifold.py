from __future__ import annotations

import torch


def _svd_left(Z: torch.Tensor, r: int) -> torch.Tensor:
    """Robust top-r left singular vectors of Z. Stays in Z's dtype (float32 on GPU --
    fp64 SVD via cuSOLVER is very slow); escalates only on failure: jittered retry,
    then a float64-on-CPU last resort for genuinely degenerate evidence."""
    Zc = Z.detach()
    if not torch.isfinite(Zc).all():          # cuSOLVER XIDs on non-finite input (uncatchable)
        Zc = torch.nan_to_num(Zc)
    try:
        U, _, _ = torch.linalg.svd(Zc, full_matrices=False)
        return U[:, :r].contiguous()
    except Exception:
        pass
    try:
        eps = 1e-6 * (float(Zc.abs().mean()) + 1e-12)
        U, _, _ = torch.linalg.svd(Zc + eps * torch.randn_like(Zc), full_matrices=False)
        return U[:, :r].contiguous()
    except Exception:
        U, _, _ = torch.linalg.svd(Zc.double().cpu(), full_matrices=False)
        return U[:, :r].to(Z.dtype).to(Z.device).contiguous()


def _ridge(G: torch.Tensor) -> torch.Tensor:
    """Add a tiny multiple of the identity to a symmetric Gram before eigh. Shifts all
    eigenvalues by eps (eigenvectors unchanged), which keeps eigh stable on rank-
    deficient / repeated-eigenvalue evidence without touching the result directions."""
    eps = 1e-6 * float(G.diagonal().mean().clamp_min(1e-30))
    return G + eps * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)


def top_left_vectors(Z: torch.Tensor, r: int) -> torch.Tensor:
    """Top-r left singular vectors of thin Z == top-r eigvecs of Z Z^T, via the small
    k x k Gram -- IN Z's DTYPE (float32 on GPU) for speed.

    The Gram eigendecomposition can fail (repeated eigenvalues from near-identical
    evidence) and RAISE; we fall back to a robust SVD on any failure or a
    numerically-dead r-th eigenvalue, so a degenerate layer never crashes the run.
    Only that fallback ever touches float64 (and only on CPU as a last resort)."""
    if not torch.isfinite(Z).all():           # cuSOLVER XIDs on non-finite input (uncatchable)
        Z = torch.nan_to_num(Z)
    m, k = Z.shape
    if min(m, k) <= r:                                 # too small -> direct SVD
        return _svd_left(Z, r)
    if m <= k:
        # SHORT-FAT (e.g. GQA v_proj, d_out < n*r): the m x m Gram Z Z^T is smaller
        # than Z^T Z, and its top-r eigenvectors ARE the left singular vectors
        # directly -- no SVD, no ill-conditioned division. This is the fix for the
        # slow v_proj path.
        G = _ridge(Z @ Z.T)
        try:
            lam, W = torch.linalg.eigh(G)
        except Exception:
            return _svd_left(Z, r)
        if float(lam[-1]) <= 0:
            return _svd_left(Z, r)
        return W[:, -r:].flip(1).contiguous()
    # TALL (m > k): the k x k Gram Z^T Z, mapped back to the left space.
    G = _ridge(Z.T @ Z)
    try:
        lam, W = torch.linalg.eigh(G)
    except Exception:
        return _svd_left(Z, r)
    lam, W = lam[-r:].flip(0), W[:, -r:].flip(1)
    if float(lam[-1]) <= 1e-8 * float(lam[0].clamp_min(1e-30)):
        return _svd_left(Z, r)
    U = (Z @ W) / lam.clamp_min(1e-30).sqrt().unsqueeze(0)
    return U.contiguous()


def grassmann_map(Ys, U_prev, V_prev, kappa_u, kappa_v, r, j_max=1, tol=1e-6):
    """argmin over (P^U, P^V) of
        sum_i ||Y_i - P^U Y_i P^V||_F^2 + (k_u/2)||P^U-P^U_prev||^2 + (k_v/2)||P^V-P^V_prev||^2 .
    With P^V fixed the U-problem is max tr[P^U (sum_i Y_i P^V Y_i^T + k_u P^U_prev)],
    solved by top-r eigenvectors; alternating is monotone. Warm-started at the
    previous bases. Returns (U, V, iterations)."""
    U, V = U_prev.clone(), V_prev.clone()
    su, sv = kappa_u ** 0.5, kappa_v ** 0.5
    Ystack = torch.stack(Ys)                      # (n, d_out, d_in): batch the matmuls
    Yt = Ystack.transpose(1, 2)                   # (n, d_in, d_out)
    d_out, d_in = Ystack.shape[1], Ystack.shape[2]
    j = 0
    for j in range(1, j_max + 1):
        U_old, V_old = U, V
        YV = torch.matmul(Ystack, V)             # (n, d_out, r): one bmm, not n matmuls
        cols = YV.permute(1, 0, 2).reshape(d_out, -1)
        if su > 0:
            cols = torch.cat([cols, su * U_prev], dim=1)
        U = top_left_vectors(cols, r)
        YtU = torch.matmul(Yt, U)                # (n, d_in, r)
        cols = YtU.permute(1, 0, 2).reshape(d_in, -1)
        if sv > 0:
            cols = torch.cat([cols, sv * V_prev], dim=1)
        V = top_left_vectors(cols, r)
        if (projector_distance(U, U_old) < tol
                and projector_distance(V, V_old) < tol):
            break
    return U, V, j


def projector_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    """||X X^T - Y Y^T||_F for orthonormal X, Y, in float64 (difference of
    near-equal numbers under a sqrt)."""
    r = X.shape[1]
    cross = float((X.double().T @ Y.double()).pow(2).sum())
    return max(2.0 * r - 2.0 * cross, 0.0) ** 0.5


def atom_similarity(U_prev, U_new, V_prev, V_new) -> torch.Tensor:
    """M[k,j] = |<u_prev_k v_prev_k^T, u_new_j v_new_j^T>_F|
             = |(U_prev^T U_new)_{k,j} * (V_prev^T V_new)_{k,j}| -- elementwise
    product of two r x r Gram matrices, so no (d_out, d_in) atom is ever formed."""
    return ((U_prev.T @ U_new) * (V_prev.T @ V_new)).abs()


def assign(M: torch.Tensor) -> torch.Tensor:
    """argmax over permutations of sum_k M[k, pi(k)]: a linear assignment, O(r^3)
    via Hungarian (scipy), greedy fallback without scipy."""
    M = torch.nan_to_num(M)                    # a NaN in M would break the assignment
    r = M.shape[0]
    try:
        from scipy.optimize import linear_sum_assignment
        _, cols = linear_sum_assignment(M.detach().cpu().numpy(), maximize=True)
        return torch.as_tensor(cols, dtype=torch.long)
    except ImportError:
        work = M.clone()
        out = torch.empty(r, dtype=torch.long)
        for _ in range(r):
            flat = int(torch.argmax(work))
            k, j = divmod(flat, r)
            out[k] = j
            work[k, :] = -1.0
            work[:, j] = -1.0
        return out