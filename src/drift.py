"""
Semantic Data Drift Analysis (SDDA) using SHAP attributions and Mann-Whitney U test
"""

from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import shap
from xgboost import XGBClassifier


def rank_biserial_effect_size(n1: int, n2: int, u_stat: float) -> float:
    """Computes Rank-Biserial correlation effect size from Mann-Whitney U statistic."""
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


class SemanticDriftDetector:
    """
    Instance-level Semantic Data Drift Detector using SHAP value distributions.
    """
    def __init__(self, base_model: XGBClassifier, X_baseline: pd.DataFrame, features: List[str]):
        self.features = features
        self.explainer = shap.TreeExplainer(base_model)
        self.baseline_shap = pd.DataFrame(
            self.explainer.shap_values(X_baseline[features]),
            columns=features
        )
        self.baseline_mean = self.baseline_shap.mean()
        self.baseline_std = self.baseline_shap.std().replace(0, 1e-10)

    def explain_dataframe(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Computes SHAP values for an input DataFrame."""
        return pd.DataFrame(
            self.explainer.shap_values(df_features[self.features]),
            columns=self.features
        )

    def compute_feature_effect_sizes(self, window_shap_df: pd.DataFrame) -> Dict[str, float]:
        """Calculates effect sizes per feature between baseline and window SHAP distributions."""
        n1 = len(self.baseline_shap)
        n2 = len(window_shap_df)
        sizes = {}
        for feature in self.features:
            stat, _ = mannwhitneyu(
                self.baseline_shap[feature],
                window_shap_df[feature],
                alternative="two-sided"
            )
            sizes[feature] = abs(rank_biserial_effect_size(n1, n2, stat))
        return sizes

    def compute_instance_drift(self, shap_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """
        Calculates raw D(x) = sum_j |(phi_j(x) - mu_j) / sigma_j| * w_j
        """
        z_scores = (shap_df - self.baseline_mean) / self.baseline_std
        w_series = pd.Series(weights)[self.features]
        return z_scores.abs().mul(w_series, axis=1).sum(axis=1)
