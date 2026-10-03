"""
Phase 5b — Full-range deferral-F1 sweep across budgets (5% to 50%)
Reproduces the Phase 2-4 pipeline exactly, then evaluates all five
deferral methods at each budget level. Reports honestly whether the
interaction score wins at any budget.
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from scipy.stats import mannwhitneyu
import shap
import os

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Bhushan\L2D with semantic drift analysis"
os.chdir(PROJECT_DIR)

# ── Rebuild Phase 2-4 pipeline (identical to Phase 5 cell 1) ─────────
print("Rebuilding Phase 2-4 pipeline...")
df = pd.read_csv("churnData_Fixed.csv")
leak_cols = ["signup_date", "year_month", "window_id"]
model_features = [c for c in df.columns if c not in leak_cols + ["Churn"]]
N_WINDOWS = df["window_id"].nunique()

window0 = df[df["window_id"] == 0]
X_window0 = window0[model_features]
y_window0 = window0["Churn"]
neg_count = (y_window0 == 0).sum()
pos_count = (y_window0 == 1).sum()
scale_pos_weight = neg_count / pos_count

u_model = CalibratedClassifierCV(
    XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                  eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight),
    method="sigmoid", cv=5,
)
u_model.fit(X_window0, y_window0)

window1 = df[df["window_id"] == 1]
X_window1 = window1[model_features]
y_window1 = window1["Churn"]
u_window1 = u_model.predict_proba(X_window1)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_window1, u_window1)
f1_scores_thresh = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
DECISION_THRESHOLD = thresholds[f1_scores_thresh[:-1].argmax()]

shap_base_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight)
shap_base_model.fit(X_window0, y_window0)
explainer = shap.TreeExplainer(shap_base_model)

shap_values_by_window = {}
for w in sorted(df["window_id"].unique()):
    window_data = df[df["window_id"] == w][model_features]
    shap_values_by_window[w] = pd.DataFrame(explainer.shap_values(window_data), columns=model_features)

baseline_shap = shap_values_by_window[0]
baseline_mean = baseline_shap.mean()
baseline_std = baseline_shap.std().replace(0, 1e-10)

def rank_biserial_effect_size(n1, n2, u_stat):
    return 1 - (2 * u_stat) / (n1 * n2)

effect_size_by_window = {}
for w in sorted(df["window_id"].unique()):
    if w == 0:
        continue
    window_shap = shap_values_by_window[w]
    sizes = {}
    for feature in model_features:
        stat, p_value = mannwhitneyu(baseline_shap[feature], window_shap[feature], alternative="two-sided")
        sizes[feature] = abs(rank_biserial_effect_size(len(baseline_shap), len(window_shap), stat))
    effect_size_by_window[w] = sizes

def compute_instance_drift_severity(window_shap_df, feature_weights):
    z_scores = (window_shap_df - baseline_mean) / baseline_std
    weights = pd.Series(feature_weights)[model_features]
    return z_scores.abs().mul(weights, axis=1).sum(axis=1)

def normalize_0_1(series):
    span = series.max() - series.min()
    if span == 0:
        return series * 0
    return (series - series.min()) / span

def get_window_u_d_error(w):
    window_data = df[df["window_id"] == w]
    X_w = window_data[model_features]
    y_w = window_data["Churn"].values
    u_w = u_model.predict_proba(X_w)[:, 1]
    d_w = normalize_0_1(compute_instance_drift_severity(shap_values_by_window[w], effect_size_by_window[w])).values
    y_pred_w = (u_w >= DECISION_THRESHOLD).astype(int)
    is_error_w = (y_pred_w != y_w).astype(int)
    return u_w, d_w, is_error_w

# Re-fit lambda
u1, d1, e1 = get_window_u_d_error(1)
u2, d2, e2 = get_window_u_d_error(2)

def deferred_by_score(score, budget):
    n = max(1, int(len(score) * budget))
    flags = np.zeros(len(score), dtype=int)
    flags[np.argsort(-score)[:n]] = 1
    return flags

best_f1, best_lambda = -1, None
for lam in np.arange(0.0, 1.05, 0.05):
    s1 = u1 + d1 + lam * u1 * d1
    s2 = u2 + d2 + lam * u2 * d2
    combined_err = np.concatenate([e1, e2])
    combined_def = np.concatenate([deferred_by_score(s1, 0.15), deferred_by_score(s2, 0.15)])
    f1 = f1_score(combined_err, combined_def, zero_division=0)
    if f1 > best_f1:
        best_f1, best_lambda = f1, lam
LAMBDA = best_lambda
print(f"Pipeline rebuilt. Decision threshold={DECISION_THRESHOLD:.3f}, lambda={LAMBDA:.2f}")

# ── Scoring functions ─────────────────────────────────────────────────

def score_uncertainty_only(u, d):
    return u

def score_drift_only(u, d):
    return d

def score_additive(u, d):
    return u + d

def score_interaction(u, d):
    return u + d + LAMBDA * u * d

def deferred_or_gate(u, d, budget):
    rank_u = pd.Series(u).rank(ascending=False, method="first")
    rank_d = pd.Series(d).rank(ascending=False, method="first")
    priority = np.minimum(rank_u.values, rank_d.values)
    n = max(1, int(len(u) * budget))
    flags = np.zeros(len(u), dtype=int)
    flags[np.argsort(priority)[:n]] = 1
    return flags

METHODS = {
    "uncertainty_only": score_uncertainty_only,
    "drift_only": score_drift_only,
    "additive": score_additive,
    "interaction": score_interaction,
}

HELD_OUT_WINDOWS = [w for w in sorted(df["window_id"].unique()) if w not in (0, 1, 2)]

# ── Full deferral-F1 sweep across budgets ─────────────────────────────
BUDGET_SWEEP = np.arange(0.05, 0.55, 0.05)

print("\n" + "="*80)
print("TASK 1: Full deferral-F1 sweep across the 5%-50% budget range")
print("="*80)

# --- Per-window sweep ---
all_results = []
for w in HELD_OUT_WINDOWS:
    u_w, d_w, err_w = get_window_u_d_error(w)
    for budget in BUDGET_SWEEP:
        for name, score_fn in METHODS.items():
            score = score_fn(u_w, d_w)
            deferred = deferred_by_score(score, budget)
            all_results.append({
                "window_id": w, "budget": round(budget, 2), "method": name,
                "precision": precision_score(err_w, deferred, zero_division=0),
                "recall": recall_score(err_w, deferred, zero_division=0),
                "f1": f1_score(err_w, deferred, zero_division=0),
            })
        # OR-gate
        deferred_or = deferred_or_gate(u_w, d_w, budget)
        all_results.append({
            "window_id": w, "budget": round(budget, 2), "method": "or_gate",
            "precision": precision_score(err_w, deferred_or, zero_division=0),
            "recall": recall_score(err_w, deferred_or, zero_division=0),
            "f1": f1_score(err_w, deferred_or, zero_division=0),
        })

sweep_df = pd.DataFrame(all_results)

# --- Average across held-out windows ---
avg_sweep = sweep_df.groupby(["budget", "method"])[["precision", "recall", "f1"]].mean().reset_index()

# --- Pivot for display ---
pivot_f1 = avg_sweep.pivot(index="budget", columns="method", values="f1")
print("\n--- Average Deferral F1 by Method x Budget (held-out windows 3-9) ---")
print(pivot_f1.round(4).to_string())

# --- Determine winner at each budget level ---
print("\n--- Winner at each budget level ---")
interaction_wins_at = []
for budget_val in sorted(pivot_f1.index):
    row = pivot_f1.loc[budget_val]
    winner = row.idxmax()
    is_interaction_win = winner == "interaction"
    marker = " <-- INTERACTION WINS" if is_interaction_win else ""
    print(f"  Budget {budget_val:.0%}: best = {winner} (F1={row.max():.4f}){marker}")
    if is_interaction_win:
        interaction_wins_at.append(budget_val)

print("\n" + "="*80)
if interaction_wins_at:
    print(f"RESULT: Interaction score wins at {len(interaction_wins_at)} budget level(s): "
          f"{[f'{b:.0%}' for b in interaction_wins_at]}")
else:
    print("RESULT: Interaction score does NOT win at any budget level in the 5%-50% range.")
print("="*80)

# --- Pooled risk and deferral-F1 ---
pooled_u_all, pooled_d_all, pooled_err_all = [], [], []
for w in HELD_OUT_WINDOWS:
    u_w, d_w, err_w = get_window_u_d_error(w)
    pooled_u_all.append(u_w)
    pooled_d_all.append(d_w)
    pooled_err_all.append(err_w)
pooled_u_all = np.concatenate(pooled_u_all)
pooled_d_all = np.concatenate(pooled_d_all)
pooled_err_all = np.concatenate(pooled_err_all)

risk_results = []
for budget in BUDGET_SWEEP:
    coverage = 1 - budget
    for name, score_fn in METHODS.items():
        score = score_fn(pooled_u_all, pooled_d_all)
        deferred = deferred_by_score(score, budget)
        accepted_mask = deferred == 0
        risk = pooled_err_all[accepted_mask].mean() if accepted_mask.sum() > 0 else np.nan
        def_f1 = f1_score(pooled_err_all, deferred, zero_division=0)
        risk_results.append({
            "budget": round(budget, 2), "coverage": round(coverage, 2),
            "method": name, "risk": risk, "deferral_f1": def_f1,
        })

    deferred_or = deferred_or_gate(pooled_u_all, pooled_d_all, budget)
    accepted_mask_or = deferred_or == 0
    risk_or = pooled_err_all[accepted_mask_or].mean() if accepted_mask_or.sum() > 0 else np.nan
    def_f1_or = f1_score(pooled_err_all, deferred_or, zero_division=0)
    risk_results.append({
        "budget": round(budget, 2), "coverage": round(coverage, 2),
        "method": "or_gate", "risk": risk_or, "deferral_f1": def_f1_or,
    })

risk_df = pd.DataFrame(risk_results)

print("\n--- Pooled Deferral F1 (held-out windows 3-9) ---")
pivot_pooled_f1 = risk_df.pivot(index="budget", columns="method", values="deferral_f1")
print(pivot_pooled_f1.round(4).to_string())

print("\n--- Pooled Residual Risk (lower = better) ---")
pivot_risk = risk_df.pivot(index="budget", columns="method", values="risk")
print(pivot_risk.round(4).to_string())

# --- Save ---
sweep_df.to_csv("phase5b_full_budget_sweep.csv", index=False)
risk_df.to_csv("phase5b_pooled_risk_f1.csv", index=False)
print("\nSaved phase5b_full_budget_sweep.csv and phase5b_pooled_risk_f1.csv")
print("\nPhase 5b complete.")
