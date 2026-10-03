"""
Phase 6 — Power BI Export & Dashboard Data Preparation

Exports:
1. Deferred-case data (customer-level U(x), D(x), Score(x), SHAP explanation, model prediction)
2. Per-window drift summary from Phase 3
3. Two dashboard views:
   a. Analyst-facing deferral queue (who to review, why, how urgent)
   b. ML-team-facing drift-trend view (which features are drifting, how fast, how much)

Output formats: CSV (Power BI can natively ingest) + JSON for richer nested data
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, f1_score
from scipy.stats import mannwhitneyu
import shap
import json
import os

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Bhushan\L2D with semantic drift analysis"
os.chdir(PROJECT_DIR)

# ── Rebuild Pipeline (identical to Phase 5) ───────────────────────────
print("Phase 6: Rebuilding pipeline...")
df = pd.read_csv("churnData_Fixed.csv")
leak_cols = ["signup_date", "year_month", "window_id"]
model_features = [c for c in df.columns if c not in leak_cols + ["Churn"]]

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

ALPHA = 0.05
effect_size_by_window = {}
drift_test_results = []  # Also save per-feature test results for drift summary
for w in sorted(df["window_id"].unique()):
    if w == 0:
        continue
    window_shap = shap_values_by_window[w]
    sizes = {}
    for feature in model_features:
        stat, p_value = mannwhitneyu(baseline_shap[feature], window_shap[feature], alternative="two-sided")
        es = abs(rank_biserial_effect_size(len(baseline_shap), len(window_shap), stat))
        sizes[feature] = es
        drift_test_results.append({
            "window_id": w, "feature": feature,
            "u_statistic": stat, "p_value": p_value,
            "effect_size": es, "significant_drift": p_value < ALPHA,
        })
    effect_size_by_window[w] = sizes

drift_tests_df = pd.DataFrame(drift_test_results)

def compute_instance_drift_severity(window_shap_df, feature_weights):
    z_scores = (window_shap_df - baseline_mean) / baseline_std
    weights = pd.Series(feature_weights)[model_features]
    return z_scores.abs().mul(weights, axis=1).sum(axis=1)

def normalize_0_1(series):
    span = series.max() - series.min()
    if span == 0:
        return series * 0
    return (series - series.min()) / span

# Re-fit lambda
def get_window_u_d_error(w):
    window_data = df[df["window_id"] == w]
    X_w = window_data[model_features]
    y_w = window_data["Churn"].values
    u_w = u_model.predict_proba(X_w)[:, 1]
    d_w = normalize_0_1(compute_instance_drift_severity(shap_values_by_window[w], effect_size_by_window[w])).values
    y_pred_w = (u_w >= DECISION_THRESHOLD).astype(int)
    is_error_w = (y_pred_w != y_w).astype(int)
    return u_w, d_w, is_error_w, y_w, y_pred_w

def deferred_by_score(score, budget):
    n = max(1, int(len(score) * budget))
    flags = np.zeros(len(score), dtype=int)
    flags[np.argsort(-score)[:n]] = 1
    return flags

u1, d1, e1, _, _ = get_window_u_d_error(1)
u2, d2, e2, _, _ = get_window_u_d_error(2)

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
DEFERRAL_BUDGET = 0.15

print(f"Pipeline rebuilt. Threshold={DECISION_THRESHOLD:.3f}, lambda={LAMBDA:.2f}")

# ═══════════════════════════════════════════════════════════════════════
# EXPORT 1: Customer-level deferral queue data
# ═══════════════════════════════════════════════════════════════════════
print("\n--- Generating customer-level deferral export ---")

customer_rows = []
for w in sorted(df["window_id"].unique()):
    if w == 0:
        continue
    window_data = df[df["window_id"] == w].copy()
    X_w = window_data[model_features]
    y_w = window_data["Churn"].values

    u_w = u_model.predict_proba(X_w)[:, 1]
    d_w_raw = compute_instance_drift_severity(shap_values_by_window[w], effect_size_by_window[w])
    d_w = normalize_0_1(d_w_raw).values

    score_w = u_w + d_w + LAMBDA * u_w * d_w

    y_pred_w = (u_w >= DECISION_THRESHOLD).astype(int)
    is_error_w = (y_pred_w != y_w).astype(int)

    budget_n = max(1, int(len(score_w) * DEFERRAL_BUDGET))
    top_k_idx = np.argsort(-score_w)[:budget_n]
    deferred_flags = np.zeros(len(score_w), dtype=int)
    deferred_flags[top_k_idx] = 1

    # Get SHAP values for this window
    window_shap = shap_values_by_window[w]

    # For each customer, find top-3 SHAP contributors (the "why flagged" explanation)
    for i in range(len(window_data)):
        shap_row = window_shap.iloc[i]
        top3_idx = shap_row.abs().nlargest(3).index.tolist()
        top3_vals = [shap_row[f] for f in top3_idx]

        # Build explanation string
        explanations = []
        for feat, val in zip(top3_idx, top3_vals):
            direction = "increases" if val > 0 else "decreases"
            explanations.append(f"{feat} ({direction} churn risk, SHAP={val:.3f})")

        customer_rows.append({
            "customer_index": window_data.index[i],
            "window_id": w,
            "predicted_churn_prob": round(u_w[i], 4),
            "model_prediction": y_pred_w[i],
            "actual_churn": y_w[i],
            "is_model_error": is_error_w[i],
            "uncertainty_score_U": round(u_w[i], 4),
            "drift_severity_D": round(d_w[i], 4),
            "composite_score": round(score_w[i], 4),
            "is_deferred": deferred_flags[i],
            "shap_explanation_1": explanations[0] if len(explanations) > 0 else "",
            "shap_explanation_2": explanations[1] if len(explanations) > 1 else "",
            "shap_explanation_3": explanations[2] if len(explanations) > 2 else "",
            # Raw feature values for context
            "tenure": X_w.iloc[i]["tenure"],
            "MonthlyCharges": X_w.iloc[i]["MonthlyCharges"],
            "TotalCharges": X_w.iloc[i]["TotalCharges"],
            "Contract_Two_year": X_w.iloc[i].get("Contract_Two year", 0),
            "Contract_One_year": X_w.iloc[i].get("Contract_One year", 0),
        })

customer_df = pd.DataFrame(customer_rows)

# Save full dataset (Power BI can filter to deferred=1)
customer_df.to_csv("phase6_customer_deferral_data.csv", index=False)
print(f"Saved phase6_customer_deferral_data.csv ({len(customer_df)} rows, "
      f"{customer_df['is_deferred'].sum()} deferred)")

# Save deferred-only subset for faster loading in the analyst queue
deferred_df = customer_df[customer_df["is_deferred"] == 1].copy()
deferred_df = deferred_df.sort_values("composite_score", ascending=False)
deferred_df.to_csv("phase6_deferred_queue.csv", index=False)
print(f"Saved phase6_deferred_queue.csv ({len(deferred_df)} deferred cases)")

# ═══════════════════════════════════════════════════════════════════════
# EXPORT 2: Per-window drift summary
# ═══════════════════════════════════════════════════════════════════════
print("\n--- Generating drift summary export ---")

# Aggregate: per-window summary
window_drift_summary = []
for w in sorted(df["window_id"].unique()):
    if w == 0:
        continue
    w_tests = drift_tests_df[drift_tests_df["window_id"] == w]
    n_sig = w_tests["significant_drift"].sum()
    avg_effect = w_tests["effect_size"].mean()
    max_effect_feat = w_tests.loc[w_tests["effect_size"].idxmax(), "feature"]
    max_effect_val = w_tests["effect_size"].max()

    # Top-5 drifting features
    top5 = w_tests.nlargest(5, "effect_size")[["feature", "effect_size", "p_value"]].to_dict("records")

    window_drift_summary.append({
        "window_id": w,
        "n_features_drifted": int(n_sig),
        "n_features_total": len(model_features),
        "pct_features_drifted": round(n_sig / len(model_features) * 100, 1),
        "avg_effect_size": round(avg_effect, 4),
        "max_effect_feature": max_effect_feat,
        "max_effect_size": round(max_effect_val, 4),
        "top5_drifting_features": json.dumps(top5, default=str),
    })

drift_summary_df = pd.DataFrame(window_drift_summary)
drift_summary_df.to_csv("phase6_drift_summary.csv", index=False)
print(f"Saved phase6_drift_summary.csv ({len(drift_summary_df)} windows)")

# Full per-feature drift data (for drill-through in Power BI)
drift_tests_df.to_csv("phase6_drift_detail.csv", index=False)
print(f"Saved phase6_drift_detail.csv ({len(drift_tests_df)} test records)")

# Feature-level trend data (effect size across windows, one row per feature per window)
feature_trend = drift_tests_df.pivot_table(
    index="feature", columns="window_id", values="effect_size"
).reset_index()
feature_trend.to_csv("phase6_feature_drift_trend.csv", index=False)
print(f"Saved phase6_feature_drift_trend.csv")

# ═══════════════════════════════════════════════════════════════════════
# EXPORT 3: Dashboard layout specifications as JSON
# ═══════════════════════════════════════════════════════════════════════
print("\n--- Generating dashboard layout specs ---")

dashboard_spec = {
    "analyst_deferral_queue": {
        "title": "Churn Deferral Queue — Analyst View",
        "description": "Customer-level queue of cases the model is uncertain about or that show explanation drift. Sorted by composite risk score.",
        "data_source": "phase6_deferred_queue.csv",
        "layout": {
            "filters": [
                {"field": "window_id", "type": "dropdown", "label": "Time Window"},
                {"field": "predicted_churn_prob", "type": "range_slider", "label": "Churn Probability", "min": 0, "max": 1},
                {"field": "drift_severity_D", "type": "range_slider", "label": "Drift Severity", "min": 0, "max": 1},
            ],
            "kpi_cards": [
                {"metric": "total_deferred", "label": "Cases in Queue", "calculation": "COUNT(*)"},
                {"metric": "avg_composite_score", "label": "Avg Risk Score", "calculation": "AVG(composite_score)"},
                {"metric": "pct_model_errors", "label": "% Model Errors (actual)", "calculation": "AVG(is_model_error) * 100"},
                {"metric": "avg_churn_prob", "label": "Avg Churn Probability", "calculation": "AVG(predicted_churn_prob)"},
            ],
            "main_table": {
                "columns": [
                    "customer_index", "window_id", "composite_score",
                    "predicted_churn_prob", "drift_severity_D",
                    "shap_explanation_1", "shap_explanation_2", "shap_explanation_3",
                    "tenure", "MonthlyCharges", "Contract_Two_year",
                ],
                "sort": {"field": "composite_score", "direction": "descending"},
                "conditional_formatting": {
                    "composite_score": {"color_scale": "red_yellow_green_reversed"},
                    "drift_severity_D": {"color_scale": "orange_gradient"},
                },
            },
            "charts": [
                {
                    "type": "scatter",
                    "title": "U(x) vs D(x) — Deferred Cases",
                    "x": "uncertainty_score_U",
                    "y": "drift_severity_D",
                    "color": "is_model_error",
                    "tooltip": ["customer_index", "shap_explanation_1"],
                },
                {
                    "type": "bar",
                    "title": "Deferred Cases by Window",
                    "x": "window_id",
                    "y": "COUNT(*)",
                    "color": "is_model_error",
                },
            ],
        },
    },
    "ml_team_drift_monitor": {
        "title": "Explanation Drift Monitor — ML Team View",
        "description": "Tracks how SHAP explanation distributions shift over time relative to the baseline window. Helps decide when to retrain.",
        "data_sources": [
            "phase6_drift_summary.csv",
            "phase6_drift_detail.csv",
            "phase6_feature_drift_trend.csv",
        ],
        "layout": {
            "filters": [
                {"field": "feature", "type": "multi_select", "label": "Features"},
                {"field": "window_id", "type": "range_slider", "label": "Window Range"},
            ],
            "kpi_cards": [
                {"metric": "latest_n_drifted", "label": "Features Drifted (Latest Window)"},
                {"metric": "latest_avg_effect", "label": "Avg Effect Size (Latest)"},
                {"metric": "retrain_signal", "label": "Retrain Signal",
                 "calculation": "IF(latest_pct_drifted > 60, 'HIGH', IF(latest_pct_drifted > 40, 'MEDIUM', 'LOW'))"},
            ],
            "charts": [
                {
                    "type": "heatmap",
                    "title": "Feature × Window Drift Heatmap",
                    "data": "phase6_drift_detail.csv",
                    "x": "window_id",
                    "y": "feature",
                    "value": "effect_size",
                    "color_scale": "blues",
                },
                {
                    "type": "line",
                    "title": "Drift Trend: Top Features Over Time",
                    "data": "phase6_feature_drift_trend.csv",
                    "x": "window_id",
                    "y": "effect_size",
                    "series": "feature",
                    "top_n": 5,
                },
                {
                    "type": "bar",
                    "title": "Number of Significantly Drifted Features per Window",
                    "data": "phase6_drift_summary.csv",
                    "x": "window_id",
                    "y": "n_features_drifted",
                },
                {
                    "type": "stacked_bar",
                    "title": "Drift Significance Breakdown per Window",
                    "data": "phase6_drift_detail.csv",
                    "x": "window_id",
                    "y": "COUNT(*)",
                    "stack": "significant_drift",
                },
            ],
        },
    },
}

with open("phase6_dashboard_spec.json", "w") as f:
    json.dump(dashboard_spec, f, indent=2)
print("Saved phase6_dashboard_spec.json")

print("\n" + "="*60)
print("Phase 6 complete. Files ready for Power BI import:")
print("  1. phase6_customer_deferral_data.csv  (full customer data)")
print("  2. phase6_deferred_queue.csv          (deferred-only, sorted)")
print("  3. phase6_drift_summary.csv           (per-window drift KPIs)")
print("  4. phase6_drift_detail.csv            (per-feature drift tests)")
print("  5. phase6_feature_drift_trend.csv     (feature trend pivot)")
print("  6. phase6_dashboard_spec.json         (dashboard layout specs)")
print("="*60)
