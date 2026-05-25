"""
AUSS metric suite for geometric analysis of steering vectors in activation space.

All metrics operate on batch_dirs: Tensor[B, d] of unit-normalised float32 vectors.

Coherence metrics (higher = model knows the topic):
  - Centroid_Norm: ||mean(batch_dirs)||_2
  - Global_Coherence: mean_i cos(v_i, mean_direction)
  - Batch_SecondMoment_TopEig: λ_max(S^T S / n)

Dispersion metrics (higher = model does NOT know the topic):
  - AUSS_L2: mean_{i≠j}(2 - 2*cos(v_i, v_j))
  - AUSS_Cos2: 1 - mean_{i≠j}(cos²(v_i, v_j))
  - AUSS_Jac: Jaccard distance of Top-32 indices
  - Batch_Cov_Trace: Tr(covariance matrix)
  - Batch_Cov_TopEig: λ_max(covariance matrix)

Returns:
  Dict of 8 scalar metrics, or NaN sentinel dict if batch_dirs is None/degenerate.
"""

import torch
import math
from typing import Dict, Optional


def centroid_norm(batch_dirs: torch.Tensor) -> float:
    """
    Coherence metric: ||mean(batch_dirs)||_2

    High value: steering vectors point in consistent direction (model knows topic)
    Low value: steering vectors are scattered/isotropic (model doesn't know topic)

    Range: [0, 1] approximately
    """
    mu = batch_dirs.mean(dim=0)
    return float(mu.norm().item())


def global_coherence(batch_dirs: torch.Tensor) -> float:
    """
    Coherence metric: mean_i cos(v_i, μ̂) where μ̂ = mean(V) / ||mean(V)||

    High value: all vectors aligned with the global mean direction
    Low value: vectors point in different directions

    Range: [-1, 1]
    """
    mu = batch_dirs.mean(dim=0)
    mu_norm = mu.norm()
    if mu_norm < 1e-8:
        return 0.0
    g_hat = mu / mu_norm
    cos_vals = batch_dirs @ g_hat  # [B]
    return float(cos_vals.mean().item())


def auss_l2(batch_dirs: torch.Tensor) -> float:
    """
    Dispersion metric: mean_{i≠j} ||v̂_i - v̂_j||²

    Equivalently: mean_{i≠j}(2 - 2*cos(v_i, v_j))

    High value: vectors point in different directions (fragmented)
    Low value: vectors point in similar directions (coherent)

    Range: [0, 2]
      0 = all vectors identical (perfect alignment)
      2 = vectors maximally dispersed (antipodal)
    """
    n = batch_dirs.shape[0]
    if n < 2:
        return 0.0

    # Compute pairwise cosines: [n, n]
    cos_mat = (batch_dirs @ batch_dirs.T).clamp(-1, 1)

    # Exclude diagonal
    mask = ~torch.eye(n, dtype=torch.bool, device=cos_mat.device)

    # Mean pairwise L2 distance
    return float((2.0 - 2.0 * cos_mat)[mask].mean().item())


def auss_cos2(batch_dirs: torch.Tensor) -> float:
    """
    Dispersion metric: 1 - mean_{i≠j} cos²(v_i, v_j)

    High value: vectors are orthogonal to each other (fragmented)
    Low value: vectors are parallel (coherent)

    Range: [0, 1]
      0 = all vectors parallel (cos² close to 1)
      1 = all vectors orthogonal (cos² close to 0)
    """
    n = batch_dirs.shape[0]
    if n < 2:
        return 0.0

    cos_mat = (batch_dirs @ batch_dirs.T).clamp(-1, 1)
    mask = ~torch.eye(n, dtype=torch.bool, device=cos_mat.device)

    return float((1.0 - cos_mat[mask].pow(2)).mean().item())


def auss_jaccard(batch_dirs: torch.Tensor, k: int = 32) -> float:
    """
    Dispersion metric: mean_{i≠j} (1 - |TopK(i) ∩ TopK(j)| / |TopK(i) ∪ TopK(j)|)

    For each batch direction vector, identify the top-k indices of largest absolute components.
    Jaccard distance measures how different these index sets are.

    High value: different neurons active in different batches (fragmented)
    Low value: same neurons active across batches (coherent)

    Range: [0, 1]
      0 = all top-k sets identical
      1 = all top-k sets disjoint
    """
    n, d = batch_dirs.shape
    if n < 2:
        return 0.0

    k_actual = min(k, d)

    # Get top-k indices for each batch direction
    _, idx = batch_dirs.abs().topk(k_actual, dim=1)
    sets = [set(row.tolist()) for row in idx]

    # Pairwise Jaccard distances
    vals = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            union_size = len(sets[i] | sets[j])
            intersection_size = len(sets[i] & sets[j])
            jaccard_sim = intersection_size / union_size if union_size > 0 else 0.0
            vals.append(1.0 - jaccard_sim)

    return float(sum(vals) / len(vals)) if vals else 0.0


def moment_metrics(batch_dirs: torch.Tensor) -> Dict[str, float]:
    """
    Compute second-moment and covariance-based metrics from batch directions.

    Returns dict with 3 metrics:
      - Batch_SecondMoment_TopEig: λ_max(S^T S / n) — coherence (high = known)
      - Batch_Cov_Trace: Tr(Σ) — total variance
      - Batch_Cov_TopEig: λ_max(Σ) — principal variance

    Range:
      All are [0, ∞)
    """
    S = batch_dirs.float()
    n, d = S.shape

    if n < 2:
        return {
            "Batch_SecondMoment_TopEig": 0.0,
            "Batch_Cov_Trace": 0.0,
            "Batch_Cov_TopEig": 0.0,
        }

    # Mean and moment matrices
    mu = S.mean(dim=0)
    M2 = (S.T @ S) / float(n)
    Sigma = M2 - torch.outer(mu, mu)

    # Symmetrise for numerical stability (eigendecomposition requires Hermitian)
    M2 = 0.5 * (M2 + M2.T)
    Sigma = 0.5 * (Sigma + Sigma.T)

    def safe_top_eig(mat: torch.Tensor) -> float:
        """Safely compute top eigenvalue, handling numerical errors."""
        try:
            eigs = torch.linalg.eigvalsh(mat)
            val = eigs[-1]  # Largest eigenvalue
            # Handle NaN/Inf
            val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
            return float(val.item())
        except Exception:
            return 0.0

    cov_trace = float(torch.nan_to_num(torch.trace(Sigma), nan=0.0).item())

    return {
        "Batch_SecondMoment_TopEig": safe_top_eig(M2),
        "Batch_Cov_Trace": cov_trace,
        "Batch_Cov_TopEig": safe_top_eig(Sigma),
    }


def compute_all_metrics(batch_dirs: Optional[torch.Tensor]) -> Dict[str, float]:
    """
    Compute all 8 AUSS metrics for a set of batch direction vectors.

    Args:
        batch_dirs: Tensor[B, d] of unit-normalised float32 vectors,
                    or None if batches are degenerate

    Returns:
        Dict with 8 keys: Centroid_Norm, Global_Coherence, AUSS_L2, AUSS_Cos2,
        AUSS_Jac, Batch_SecondMoment_TopEig, Batch_Cov_Trace, Batch_Cov_TopEig

        If batch_dirs is None or B < 2: returns dict of NaN for all metrics
    """
    if batch_dirs is None or batch_dirs.shape[0] < 2:
        return {
            "Centroid_Norm": float("nan"),
            "Global_Coherence": float("nan"),
            "AUSS_L2": float("nan"),
            "AUSS_Cos2": float("nan"),
            "AUSS_Jac": float("nan"),
            "Batch_SecondMoment_TopEig": float("nan"),
            "Batch_Cov_Trace": float("nan"),
            "Batch_Cov_TopEig": float("nan"),
        }

    out = {}
    out["Centroid_Norm"] = centroid_norm(batch_dirs)
    out["Global_Coherence"] = global_coherence(batch_dirs)
    out["AUSS_L2"] = auss_l2(batch_dirs)
    out["AUSS_Cos2"] = auss_cos2(batch_dirs)
    out["AUSS_Jac"] = auss_jaccard(batch_dirs, k=32)
    out.update(moment_metrics(batch_dirs))

    return out
