"""
Gram-matrix entropy metrics for raw hidden-state and anon-diff vector batches.

Implements the matrix-based entropy from:
  Skean et al. (2025) "Layer by Layer: Uncovering Hidden Representations in LMs"
  arXiv:2502.02013

Two functions:
  gram_entropy(vecs)       — S_alpha (Rényi order 2 + Von Neumann) + EffRank
  raw_vector_stats(vecs)   — mean L2 norm + mean pairwise cosine

Both operate on raw [N, d] float tensors (not unit-normalised, not batch-averaged).
"""

import math
import torch
from typing import Dict


_NAN_RESULT = {
    "S_alpha_2": float("nan"),
    "S_alpha_1": float("nan"),
    "EffRank": float("nan"),
}

_NAN_STATS = {
    "MeanL2": float("nan"),
    "MeanCos": float("nan"),
}


def gram_entropy(vecs: torch.Tensor) -> Dict[str, float]:
    """
    Gram-matrix entropy on a raw [N, d] float tensor.

    K_Z = ZZ^T / tr(ZZ^T)                       (normalised Gram matrix)
    S_2(Z) = -log2(tr(K_Z^2))                   (Rényi order-2, stable)
    S_1(Z) = -sum_i lambda_i * log2(lambda_i)   (Von Neumann, alpha -> 1)
    EffRank = 2^S_1                              (effective rank in linear scale)

    Lower entropy = more compressed = model knows the concept.
    Higher entropy = more uniform eigenvalue spread = fragmented / unknown.

    Args:
        vecs: [N, d] float tensor of raw (unnormalised) vectors.

    Returns:
        Dict with keys: S_alpha_2, S_alpha_1, EffRank.
        All NaN if N < 2 or tr(ZZ^T) < 1e-10.
    """
    if vecs is None or vecs.ndim != 2 or vecs.shape[0] < 2:
        return dict(_NAN_RESULT)

    Z = vecs.float()
    G = Z @ Z.T  # [N, N] Gram matrix

    trace_G = G.trace()
    if trace_G.item() < 1e-10 or not torch.isfinite(trace_G):
        return dict(_NAN_RESULT)

    K = G / trace_G  # Normalised Gram [N, N]
    K = 0.5 * (K + K.T)  # Symmetrise for numerical stability

    try:
        eigs = torch.linalg.eigvalsh(K)  # Ascending order, [N]
    except Exception:
        return dict(_NAN_RESULT)

    eigs = eigs.clamp(min=0.0)
    eig_sum = eigs.sum()
    if eig_sum.item() < 1e-12:
        return dict(_NAN_RESULT)
    eigs = eigs / eig_sum  # Renormalise to sum=1

    # S_2 = -log2(sum lambda_i^2)
    s2_val = -(eigs.pow(2).sum() + 1e-12).log2()
    s2 = float(torch.nan_to_num(s2_val, nan=float("nan")).item())

    # S_1 = -sum lambda_i * log2(lambda_i)  (skip near-zero eigenvalues)
    nonzero = eigs[eigs > 1e-12]
    if nonzero.numel() == 0:
        return dict(_NAN_RESULT)
    s1_val = -(nonzero * nonzero.log2()).sum()
    s1 = float(torch.nan_to_num(s1_val, nan=float("nan")).item())

    # EffRank = 2^S_1  (convert from bits to linear scale)
    try:
        eff_rank = math.pow(2.0, s1) if math.isfinite(s1) else float("nan")
    except (OverflowError, ValueError):
        eff_rank = float("nan")

    return {"S_alpha_2": s2, "S_alpha_1": s1, "EffRank": eff_rank}


def raw_vector_stats(vecs: torch.Tensor) -> Dict[str, float]:
    """
    Mean L2 norm and mean pairwise cosine similarity on raw [N, d] vectors.

    MeanL2  = (1/N) sum_i ||v_i||_2
    MeanCos = mean_{i != j} cos(v_i, v_j)

    Args:
        vecs: [N, d] float tensor of raw (unnormalised) vectors.

    Returns:
        Dict with keys: MeanL2, MeanCos. Both NaN if N < 2.
    """
    if vecs is None or vecs.ndim != 2 or vecs.shape[0] < 2:
        return dict(_NAN_STATS)

    Z = vecs.float()
    norms = Z.norm(dim=1)  # [N]
    mean_l2 = float(norms.mean().item())

    # Normalise for cosine (guard against zero-norm rows)
    safe_norms = norms.clamp(min=1e-12).unsqueeze(1)
    Z_norm = Z / safe_norms  # [N, d]

    cos_mat = Z_norm @ Z_norm.T  # [N, N], range [-1, 1]
    n = Z.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=cos_mat.device)
    mean_cos = float(cos_mat[mask].mean().item())

    return {"MeanL2": mean_l2, "MeanCos": mean_cos}


def compute_entropy_metrics(
    reg_vecs: torch.Tensor,
    anon_vecs: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all entropy metrics for one layer given paired reg/anon vectors.

    Operates on:
      - reg_vecs  [N, d]: raw regular hidden states
      - anon_diff [N, d]: reg - anon (concept-differential, unnormalised)
      - anon_diff_norm [N, d]: anon_diff row-normalised to unit length

    Returns flat dict with prefixed keys:
      Reg_S_alpha_2, Reg_S_alpha_1, Reg_EffRank, Reg_MeanL2, Reg_MeanCos
      AnonDiff_S_alpha_2, AnonDiff_S_alpha_1, AnonDiff_EffRank, AnonDiff_MeanL2, AnonDiff_MeanCos
      NormAnonDiff_S_alpha_2, NormAnonDiff_S_alpha_1, NormAnonDiff_EffRank, NormAnonDiff_MeanL2, NormAnonDiff_MeanCos
    """
    anon_diff = reg_vecs.float() - anon_vecs.float()

    # Row-normalise anon-diff to isolate directional structure from magnitude
    norms = anon_diff.norm(dim=1, keepdim=True).clamp(min=1e-12)
    anon_diff_norm = anon_diff / norms

    reg_ent   = gram_entropy(reg_vecs)
    reg_stats = raw_vector_stats(reg_vecs)

    diff_ent   = gram_entropy(anon_diff)
    diff_stats = raw_vector_stats(anon_diff)

    norm_ent   = gram_entropy(anon_diff_norm)
    norm_stats = raw_vector_stats(anon_diff_norm)

    result = {}
    for k, v in reg_ent.items():   result[f"Reg_{k}"]          = v
    for k, v in reg_stats.items(): result[f"Reg_{k}"]          = v
    for k, v in diff_ent.items():  result[f"AnonDiff_{k}"]     = v
    for k, v in diff_stats.items():result[f"AnonDiff_{k}"]     = v
    for k, v in norm_ent.items():  result[f"NormAnonDiff_{k}"] = v
    for k, v in norm_stats.items():result[f"NormAnonDiff_{k}"] = v

    return result
