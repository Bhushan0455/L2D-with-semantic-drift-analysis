"""
Cost-Aware Deferral Policy and Composite Scoring
"""

import numpy as np
import pandas as pd


def compute_composite_score(
    u_scores: np.ndarray,
    d_scores: np.ndarray,
    lambda_param: float = 0.15
) -> np.ndarray:
    """
    Computes composite interaction score:
    Score(x) = U(x) + D(x) + lambda * U(x) * D(x)
    """
    return u_scores + d_scores + lambda_param * (u_scores * d_scores)


def apply_deferral_budget(scores: np.ndarray, budget: float = 0.15) -> np.ndarray:
    """
    Assigns top (budget * N) highest score samples to be deferred (is_deferred = 1).
    """
    n_defer = max(1, int(len(scores) * budget))
    deferred_flags = np.zeros(len(scores), dtype=int)
    top_indices = np.argsort(-scores)[:n_defer]
    deferred_flags[top_indices] = 1
    return deferred_flags
