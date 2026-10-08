"""
Phase 6 — Interactive Model Testing UI
Flask web app for testing the L2D churn model interactively.

Features:
  1. Single-customer prediction with SHAP explanation
  2. Model accuracy dashboard (confusion matrix, per-window metrics)
  3. Customer explorer (browse existing predictions, filter/sort)

Run:  python Phase6_interactive_app.py
Then:  open http://localhost:5000
"""

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template_string
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix, roc_curve
)
from scipy.stats import mannwhitneyu
import shap
import json
import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

app = Flask(__name__)

# ── Pipeline rebuild (same as Phase 5b / Phase 6) ────────────────────
print("Phase 6 Interactive: Rebuilding pipeline...")

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
DECISION_THRESHOLD = float(thresholds[f1_scores_thresh[:-1].argmax()])

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

# Global D(x) normalization
raw_d_by_window = {}
for w in sorted(df["window_id"].unique()):
    if w == 0:
        continue
    raw_d_by_window[w] = compute_instance_drift_severity(shap_values_by_window[w], effect_size_by_window[w])

all_raw_d = pd.concat([raw_d_by_window[w] for w in sorted(raw_d_by_window.keys())])
global_d_min = float(all_raw_d.min())
global_d_max = float(all_raw_d.max())
global_d_span = global_d_max - global_d_min

# Re-fit lambda with global D(x)
def get_window_u_d_error(w):
    window_data = df[df["window_id"] == w]
    X_w = window_data[model_features]
    y_w = window_data["Churn"].values
    u_w = u_model.predict_proba(X_w)[:, 1]
    d_w = ((raw_d_by_window[w] - global_d_min) / global_d_span).values
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
LAMBDA = float(best_lambda)
DEFERRAL_BUDGET = 0.15

# Pre-compute per-window accuracy metrics for the accuracy dashboard
accuracy_data = []
all_y_true = []
all_y_prob = []
all_y_pred = []

for w in sorted(df["window_id"].unique()):
    if w == 0:
        continue
    window_data = df[df["window_id"] == w]
    X_w = window_data[model_features]
    y_w = window_data["Churn"].values
    u_w = u_model.predict_proba(X_w)[:, 1]
    y_pred_w = (u_w >= DECISION_THRESHOLD).astype(int)

    all_y_true.extend(y_w.tolist())
    all_y_prob.extend(u_w.tolist())
    all_y_pred.extend(y_pred_w.tolist())

    acc = accuracy_score(y_w, y_pred_w)
    prec = precision_score(y_w, y_pred_w, zero_division=0)
    rec = recall_score(y_w, y_pred_w, zero_division=0)
    f1 = f1_score(y_w, y_pred_w, zero_division=0)
    try:
        auc = roc_auc_score(y_w, u_w)
    except:
        auc = 0
    cm = confusion_matrix(y_w, y_pred_w).tolist()
    n_total = len(y_w)
    n_churn = int(y_w.sum())
    churn_rate = round(n_churn / n_total * 100, 1)

    accuracy_data.append({
        "window_id": int(w),
        "n_customers": n_total,
        "n_churn": n_churn,
        "churn_rate": churn_rate,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "confusion_matrix": cm,
    })

# Overall metrics
all_y_true = np.array(all_y_true)
all_y_prob = np.array(all_y_prob)
all_y_pred = np.array(all_y_pred)

overall_metrics = {
    "accuracy": round(float(accuracy_score(all_y_true, all_y_pred)), 4),
    "precision": round(float(precision_score(all_y_true, all_y_pred, zero_division=0)), 4),
    "recall": round(float(recall_score(all_y_true, all_y_pred, zero_division=0)), 4),
    "f1": round(float(f1_score(all_y_true, all_y_pred, zero_division=0)), 4),
    "auc": round(float(roc_auc_score(all_y_true, all_y_prob)), 4),
    "n_total": int(len(all_y_true)),
    "n_churn": int(all_y_true.sum()),
    "confusion_matrix": confusion_matrix(all_y_true, all_y_pred).tolist(),
    "threshold": round(DECISION_THRESHOLD, 4),
    "lambda": round(LAMBDA, 2),
}

# ROC curve data
fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr[::5], tpr[::5])]
roc_points.append({"fpr": round(float(fpr[-1]), 4), "tpr": round(float(tpr[-1]), 4)})

# Calibration data (bucketed)
calib_df = pd.DataFrame({"prob": all_y_prob, "actual": all_y_true})
calib_df["bucket"] = pd.cut(calib_df["prob"], bins=np.arange(0, 1.05, 0.1))
calib_groups = calib_df.groupby("bucket", observed=True).agg(
    avg_predicted=("prob", "mean"),
    avg_actual=("actual", "mean"),
    count=("actual", "size"),
).dropna()
calibration_data = [
    {"predicted": round(row["avg_predicted"], 4), "actual": round(row["avg_actual"], 4), "count": int(row["count"])}
    for _, row in calib_groups.iterrows()
]

# Pre-compute customer data for explorer
customer_export = pd.read_csv("phase6_customer_deferral_data.csv")

print(f"Pipeline ready. threshold={DECISION_THRESHOLD:.3f}, lambda={LAMBDA:.2f}")
print(f"Overall: Accuracy={overall_metrics['accuracy']}, F1={overall_metrics['f1']}, AUC={overall_metrics['auc']}")
print(f"Serving on http://localhost:5000")


# ═══════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.route("/api/features")
def api_features():
    """Return list of model features with their ranges."""
    feature_info = []
    for f in model_features:
        vals = df[f]
        feature_info.append({
            "name": f,
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
            "mean": round(float(vals.mean()), 2),
            "is_binary": bool(vals.isin([0, 1]).all()),
        })
    return jsonify(feature_info)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict churn for a single customer with SHAP explanation."""
    data = request.json
    feature_values = {}
    for f in model_features:
        feature_values[f] = float(data.get(f, df[f].mean()))

    X_input = pd.DataFrame([feature_values])
    prob = float(u_model.predict_proba(X_input)[:, 1][0])
    pred = int(prob >= DECISION_THRESHOLD)

    # SHAP values for this customer
    shap_vals = explainer.shap_values(X_input)[0]
    base_value = float(explainer.expected_value)

    # Top features by absolute SHAP impact
    shap_ranked = sorted(zip(model_features, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    shap_explanation = [
        {"feature": f, "value": round(float(v), 4), "feature_value": round(feature_values[f], 2)}
        for f, v in shap_ranked[:10]
    ]

    # For drift, use window 9 effect sizes as a proxy for "latest"
    latest_weights = effect_size_by_window[9]
    z_scores = (pd.Series(shap_vals, index=model_features) - baseline_mean) / baseline_std
    weights = pd.Series(latest_weights)[model_features]
    raw_drift = float(z_scores.abs().mul(weights).sum())
    d_normalized = max(0, min(1, (raw_drift - global_d_min) / global_d_span))

    composite_score = prob + d_normalized + LAMBDA * prob * d_normalized

    # Determine deferral threshold (based on the 85th percentile of scores in window 9)
    # We'll use a fixed approach: compare against the deferral queue
    would_defer = composite_score >= 0.5  # approximate threshold

    return jsonify({
        "churn_probability": round(prob, 4),
        "prediction": pred,
        "prediction_label": "CHURN" if pred == 1 else "NO CHURN",
        "uncertainty_U": round(prob, 4),
        "drift_severity_D": round(d_normalized, 4),
        "composite_score": round(composite_score, 4),
        "would_defer": bool(would_defer),
        "lambda": round(LAMBDA, 2),
        "threshold": round(DECISION_THRESHOLD, 4),
        "shap_base_value": round(base_value, 4),
        "shap_explanation": shap_explanation,
        "all_shap": [{"feature": f, "value": round(float(v), 4)} for f, v in zip(model_features, shap_vals)],
    })


@app.route("/api/accuracy")
def api_accuracy():
    """Return accuracy metrics for the dashboard."""
    return jsonify({
        "overall": overall_metrics,
        "per_window": accuracy_data,
        "roc_curve": roc_points,
        "calibration": calibration_data,
    })


@app.route("/api/customers")
def api_customers():
    """Return paginated customer data for the explorer."""
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    window = request.args.get("window", "all")
    filter_type = request.args.get("filter", "all")
    sort_by = request.args.get("sort", "composite_score")
    sort_dir = request.args.get("dir", "desc")

    filtered = customer_export.copy()
    if window != "all":
        filtered = filtered[filtered["window_id"] == int(window)]
    if filter_type == "deferred":
        filtered = filtered[filtered["is_deferred"] == 1]
    elif filter_type == "errors":
        filtered = filtered[filtered["is_model_error"] == 1]
    elif filter_type == "deferred_errors":
        filtered = filtered[(filtered["is_deferred"] == 1) & (filtered["is_model_error"] == 1)]

    ascending = sort_dir == "asc"
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=ascending)

    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    page_data = filtered.iloc[start:end]

    cols = ["customer_index", "window_id", "predicted_churn_prob", "model_prediction",
            "actual_churn", "is_model_error", "uncertainty_score_U", "drift_severity_D",
            "composite_score", "is_deferred", "shap_explanation_1", "tenure", "MonthlyCharges"]
    available_cols = [c for c in cols if c in page_data.columns]

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
        "data": page_data[available_cols].to_dict(orient="records"),
    })


@app.route("/api/customer/<int:customer_id>")
def api_customer_detail(customer_id):
    """Return detailed data for a specific customer."""
    row = customer_export[customer_export["customer_index"] == customer_id]
    if len(row) == 0:
        return jsonify({"error": "Customer not found"}), 404

    row = row.iloc[0]
    w = int(row["window_id"])

    # Get raw feature values for this customer
    customer_data = df.loc[customer_id]
    feature_vals = {f: float(customer_data[f]) for f in model_features}

    # Get SHAP values
    X_input = pd.DataFrame([feature_vals])
    shap_vals = explainer.shap_values(X_input)[0]
    shap_ranked = sorted(zip(model_features, shap_vals), key=lambda x: abs(x[1]), reverse=True)

    return jsonify({
        "customer_index": int(customer_id),
        "window_id": w,
        "features": feature_vals,
        "churn_probability": round(float(row["predicted_churn_prob"]), 4),
        "prediction": int(row["model_prediction"]),
        "actual_churn": int(row["actual_churn"]),
        "is_error": int(row["is_model_error"]),
        "uncertainty_U": round(float(row["uncertainty_score_U"]), 4),
        "drift_severity_D": round(float(row["drift_severity_D"]), 4),
        "composite_score": round(float(row["composite_score"]), 4),
        "is_deferred": int(row["is_deferred"]),
        "shap_explanation": [
            {"feature": f, "value": round(float(v), 4), "feature_value": round(feature_vals[f], 2)}
            for f, v in shap_ranked[:10]
        ],
        "shap_base_value": round(float(explainer.expected_value), 4),
    })


# ═══════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L2D Churn Model — Interactive Testing</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0e14;
            --bg2: #131720;
            --bg3: #1a1f2e;
            --bg4: #222840;
            --text: #e4e7ed;
            --text2: #8e95a9;
            --text3: #5c6378;
            --blue: #3b82f6;
            --purple: #8b5cf6;
            --teal: #14b8a6;
            --orange: #f59e0b;
            --red: #ef4444;
            --green: #22c55e;
            --pink: #ec4899;
            --border: #252b3d;
            --radius: 12px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }

        /* HEADER */
        .app-header {
            background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(139,92,246,0.08));
            border-bottom: 1px solid var(--border);
            padding: 16px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .app-title {
            font-size: 18px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--blue), var(--purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .app-subtitle { font-size: 12px; color: var(--text2); margin-top: 2px; }
        .pipeline-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(34,197,94,0.1);
            border: 1px solid rgba(34,197,94,0.25);
            color: var(--green);
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 600;
        }

        /* NAVIGATION */
        .nav-tabs {
            display: flex;
            gap: 2px;
            padding: 12px 32px;
            background: var(--bg2);
            border-bottom: 1px solid var(--border);
        }
        .nav-tab {
            padding: 10px 24px;
            border: none;
            background: transparent;
            color: var(--text2);
            font-family: inherit;
            font-size: 13px;
            font-weight: 500;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .nav-tab:hover { color: var(--text); background: var(--bg3); }
        .nav-tab.active { color: white; background: var(--blue); box-shadow: 0 2px 8px rgba(59,130,246,0.3); }

        /* MAIN */
        .main { padding: 24px 32px; max-width: 1480px; margin: 0 auto; }
        .tab-panel { display: none; }
        .tab-panel.active { display: block; }

        /* CARDS */
        .card {
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 20px;
        }
        .card-title {
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* KPI */
        .kpi-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .kpi {
            background: var(--bg3);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px;
            text-align: center;
            transition: transform 0.2s;
        }
        .kpi:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
        .kpi-label { font-size: 10px; color: var(--text3); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
        .kpi-value { font-size: 32px; font-weight: 800; line-height: 1; }
        .kpi-sub { font-size: 11px; color: var(--text2); margin-top: 6px; }

        /* GRID */
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }

        /* FORM */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 12px;
        }
        .form-group { display: flex; flex-direction: column; gap: 4px; }
        .form-group label {
            font-size: 11px;
            color: var(--text2);
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        .form-group input, .form-group select {
            background: var(--bg2);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 13px;
            font-family: inherit;
            transition: border-color 0.2s;
        }
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: var(--blue);
            box-shadow: 0 0 0 3px rgba(59,130,246,0.15);
        }
        .toggle-wrap {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 0;
        }
        .toggle {
            position: relative;
            width: 42px;
            height: 22px;
            background: var(--bg2);
            border-radius: 11px;
            border: 1px solid var(--border);
            cursor: pointer;
            transition: background 0.2s;
        }
        .toggle::after {
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 16px;
            height: 16px;
            background: var(--text2);
            border-radius: 50%;
            transition: all 0.2s;
        }
        .toggle.on { background: var(--blue); border-color: var(--blue); }
        .toggle.on::after { left: 22px; background: white; }

        /* BUTTONS */
        .btn {
            padding: 12px 28px;
            border: none;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--blue), var(--purple));
            color: white;
            box-shadow: 0 2px 8px rgba(59,130,246,0.3);
        }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(59,130,246,0.4); }
        .btn-secondary {
            background: var(--bg2);
            color: var(--text);
            border: 1px solid var(--border);
        }
        .btn-secondary:hover { background: var(--bg4); }

        /* RESULT PANEL */
        .result-panel {
            background: var(--bg2);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            margin-top: 20px;
        }

        /* GAUGE */
        .gauge-container {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            padding: 16px;
        }
        .gauge-ring {
            width: 140px;
            height: 140px;
            border-radius: 50%;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .gauge-inner {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: var(--bg3);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }
        .gauge-value { font-size: 24px; font-weight: 800; }
        .gauge-label { font-size: 10px; color: var(--text2); margin-top: 4px; }
        .gauge-caption { font-size: 12px; color: var(--text2); margin-top: 8px; font-weight: 600; }

        /* SHAP BAR */
        .shap-bar-container { margin: 4px 0; display: flex; align-items: center; gap: 8px; }
        .shap-feature { width: 180px; text-align: right; font-size: 11px; color: var(--text2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .shap-bar-track { flex: 1; height: 20px; background: var(--bg2); border-radius: 4px; position: relative; overflow: hidden; }
        .shap-bar-fill { height: 100%; border-radius: 4px; position: absolute; top: 0; transition: width 0.5s ease; }
        .shap-val { width: 60px; font-size: 11px; font-weight: 600; }

        /* TABLE */
        .data-tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
        .data-tbl th {
            background: var(--bg2);
            color: var(--text3);
            text-transform: uppercase;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.5px;
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            cursor: pointer;
        }
        .data-tbl th:hover { color: var(--text); }
        .data-tbl td {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            color: var(--text2);
        }
        .data-tbl tr:hover td { background: var(--bg4); color: var(--text); }

        /* BADGE */
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-red { background: rgba(239,68,68,0.15); color: var(--red); }
        .badge-green { background: rgba(34,197,94,0.15); color: var(--green); }
        .badge-blue { background: rgba(59,130,246,0.15); color: var(--blue); }
        .badge-orange { background: rgba(245,158,11,0.15); color: var(--orange); }
        .badge-purple { background: rgba(139,92,246,0.15); color: var(--purple); }

        /* CONFUSION MATRIX */
        .cm-grid {
            display: grid;
            grid-template-columns: auto 1fr 1fr;
            grid-template-rows: auto 1fr 1fr;
            gap: 4px;
            max-width: 280px;
            margin: 0 auto;
        }
        .cm-cell {
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            font-weight: 700;
            font-size: 20px;
        }
        .cm-label {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            color: var(--text3);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .cm-corner { }
        .cm-tp { background: rgba(34,197,94,0.15); color: var(--green); }
        .cm-tn { background: rgba(59,130,246,0.1); color: var(--blue); }
        .cm-fp { background: rgba(245,158,11,0.12); color: var(--orange); }
        .cm-fn { background: rgba(239,68,68,0.12); color: var(--red); }
        .cm-sub { font-size: 10px; font-weight: 400; color: var(--text3); margin-top: 4px; }

        /* CHART CANVAS */
        .chart-wrap { position: relative; }
        canvas { width: 100% !important; height: 260px !important; }

        /* PAGINATION */
        .pagination {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-top: 16px;
        }
        .pagination button {
            padding: 6px 14px;
            border: 1px solid var(--border);
            background: var(--bg2);
            color: var(--text2);
            border-radius: 6px;
            cursor: pointer;
            font-family: inherit;
            font-size: 12px;
        }
        .pagination button:hover { background: var(--bg4); color: var(--text); }
        .pagination button.active { background: var(--blue); color: white; border-color: var(--blue); }
        .pagination span { font-size: 12px; color: var(--text2); }

        /* FILTERS */
        .filter-row {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
        }
        .filter-row select, .filter-row input {
            background: var(--bg2);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-family: inherit;
        }

        /* DEFER VERDICT */
        .verdict {
            text-align: center;
            padding: 20px;
            border-radius: var(--radius);
            margin-top: 16px;
        }
        .verdict-defer {
            background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(239,68,68,0.08));
            border: 1px solid rgba(245,158,11,0.25);
        }
        .verdict-auto {
            background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(59,130,246,0.08));
            border: 1px solid rgba(34,197,94,0.25);
        }
        .verdict-icon { font-size: 36px; margin-bottom: 8px; }
        .verdict-title { font-size: 16px; font-weight: 700; margin-bottom: 4px; }
        .verdict-sub { font-size: 12px; color: var(--text2); }

        @media (max-width: 900px) {
            .grid-2, .grid-3 { grid-template-columns: 1fr; }
            .main { padding: 16px; }
        }

        .loading { text-align: center; padding: 40px; color: var(--text3); }
        .loading::after { content: '⟳'; animation: spin 1s linear infinite; display: inline-block; font-size: 24px; margin-left: 8px; }
        @keyframes spin { 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>

<div class="app-header">
    <div>
        <div class="app-title">L2D Churn Model — Interactive Testing</div>
        <div class="app-subtitle">Learning to Defer with Semantic Drift Analysis • XGBoost + Platt Calibration + SHAP Drift</div>
    </div>
    <div class="pipeline-badge">✓ Pipeline Active • λ = """ + str(round(LAMBDA, 2)) + """ • threshold = """ + str(round(DECISION_THRESHOLD, 3)) + """</div>
</div>

<div class="nav-tabs">
    <button class="nav-tab active" onclick="switchTab('predict')">🔮 Predict</button>
    <button class="nav-tab" onclick="switchTab('accuracy')">📊 Model Accuracy</button>
    <button class="nav-tab" onclick="switchTab('explorer')">🔍 Customer Explorer</button>
</div>

<div class="main">

    <!-- ═══════════ TAB 1: PREDICT ═══════════ -->
    <div class="tab-panel active" id="tab-predict">
        <div class="card">
            <div class="card-title">📝 Customer Feature Input</div>
            <p style="font-size:12px; color:var(--text2); margin-bottom:16px;">Enter customer attributes below. Binary features use toggles (0/1). Numeric features have sliders with value inputs.</p>
            <div id="feature-form" class="form-grid">
                <div class="loading">Loading feature definitions...</div>
            </div>
            <div style="margin-top: 20px; display: flex; gap: 12px; align-items: center;">
                <button class="btn btn-primary" onclick="runPrediction()">⚡ Run Prediction</button>
                <button class="btn btn-secondary" onclick="loadRandomCustomer()">🎲 Load Random Customer</button>
                <button class="btn btn-secondary" onclick="resetForm()">↺ Reset to Means</button>
            </div>
        </div>

        <div id="prediction-result" style="display:none;">
            <!-- Filled by JS -->
        </div>
    </div>

    <!-- ═══════════ TAB 2: ACCURACY ═══════════ -->
    <div class="tab-panel" id="tab-accuracy">
        <div id="accuracy-content">
            <div class="loading">Loading accuracy metrics...</div>
        </div>
    </div>

    <!-- ═══════════ TAB 3: EXPLORER ═══════════ -->
    <div class="tab-panel" id="tab-explorer">
        <div class="card">
            <div class="card-title">🔍 Customer Data Explorer</div>
            <div class="filter-row">
                <select id="exp-window" onchange="loadExplorerPage(1)">
                    <option value="all">All Windows</option>
                    <option value="1">Window 1</option><option value="2">Window 2</option>
                    <option value="3">Window 3</option><option value="4">Window 4</option>
                    <option value="5">Window 5</option><option value="6">Window 6</option>
                    <option value="7">Window 7</option><option value="8">Window 8</option>
                    <option value="9">Window 9</option>
                </select>
                <select id="exp-filter" onchange="loadExplorerPage(1)">
                    <option value="all">All Cases</option>
                    <option value="deferred">Deferred Only</option>
                    <option value="errors">Model Errors Only</option>
                    <option value="deferred_errors">Deferred Errors</option>
                </select>
                <select id="exp-sort" onchange="loadExplorerPage(1)">
                    <option value="composite_score">Composite Score ↓</option>
                    <option value="predicted_churn_prob">Churn Prob ↓</option>
                    <option value="drift_severity_D">Drift D(x) ↓</option>
                </select>
            </div>
            <div id="explorer-table" style="overflow-x: auto;">
                <div class="loading">Loading customers...</div>
            </div>
            <div id="explorer-pagination" class="pagination"></div>
        </div>

        <div id="customer-detail" style="display:none;" class="card">
            <!-- Filled by JS when a customer is clicked -->
        </div>
    </div>
</div>

<script>
let featureInfo = [];
let currentValues = {};

function switchTab(tab) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + tab).classList.add('active');
    event.target.classList.add('active');
    if (tab === 'accuracy') loadAccuracy();
    if (tab === 'explorer') loadExplorerPage(1);
}

// ── Feature Form ──
async function loadFeatures() {
    const resp = await fetch('/api/features');
    featureInfo = await resp.json();
    const form = document.getElementById('feature-form');
    form.innerHTML = '';
    featureInfo.forEach(f => {
        currentValues[f.name] = f.mean;
        const grp = document.createElement('div');
        grp.className = 'form-group';
        if (f.is_binary) {
            grp.innerHTML = `<label>${f.name}</label>
                <div class="toggle-wrap">
                    <div class="toggle ${f.mean >= 0.5 ? 'on' : ''}" id="tog-${f.name}" onclick="toggleBinary('${f.name}')"></div>
                    <span id="togval-${f.name}" style="font-size:12px;">${Math.round(f.mean)}</span>
                </div>`;
        } else {
            grp.innerHTML = `<label>${f.name}</label>
                <input type="number" id="inp-${f.name}" value="${f.mean}" step="0.01"
                    min="${f.min}" max="${f.max}" onchange="currentValues['${f.name}']=parseFloat(this.value)">`;
        }
        form.appendChild(grp);
    });
}

function toggleBinary(name) {
    const el = document.getElementById('tog-' + name);
    const val = el.classList.toggle('on') ? 1 : 0;
    currentValues[name] = val;
    document.getElementById('togval-' + name).textContent = val;
}

function resetForm() {
    featureInfo.forEach(f => {
        currentValues[f.name] = f.mean;
        if (f.is_binary) {
            const el = document.getElementById('tog-' + f.name);
            if (f.mean >= 0.5) el.classList.add('on'); else el.classList.remove('on');
            document.getElementById('togval-' + f.name).textContent = Math.round(f.mean);
        } else {
            document.getElementById('inp-' + f.name).value = f.mean;
        }
    });
}

async function loadRandomCustomer() {
    const windowId = Math.floor(Math.random() * 9) + 1;
    const resp = await fetch(`/api/customers?window=${windowId}&per_page=100`);
    const data = await resp.json();
    if (data.data.length === 0) return;
    const cust = data.data[Math.floor(Math.random() * data.data.length)];
    const detail = await fetch(`/api/customer/${cust.customer_index}`);
    const d = await detail.json();
    for (const [name, val] of Object.entries(d.features)) {
        currentValues[name] = val;
        const fi = featureInfo.find(f => f.name === name);
        if (fi && fi.is_binary) {
            const el = document.getElementById('tog-' + name);
            if (val >= 0.5) el.classList.add('on'); else el.classList.remove('on');
            document.getElementById('togval-' + name).textContent = Math.round(val);
        } else {
            const inp = document.getElementById('inp-' + name);
            if (inp) inp.value = val;
        }
    }
}

// ── Prediction ──
async function runPrediction() {
    // Sync numeric inputs
    featureInfo.forEach(f => {
        if (!f.is_binary) {
            const inp = document.getElementById('inp-' + f.name);
            if (inp) currentValues[f.name] = parseFloat(inp.value);
        }
    });

    const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentValues),
    });
    const r = await resp.json();
    showPredictionResult(r);
}

function showPredictionResult(r) {
    const el = document.getElementById('prediction-result');
    el.style.display = 'block';
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });

    const probPct = (r.churn_probability * 100).toFixed(1);
    const gaugeColor = r.churn_probability > 0.5 ? 'var(--red)' : (r.churn_probability > 0.2 ? 'var(--orange)' : 'var(--green)');
    const driftColor = r.drift_severity_D > 0.7 ? 'var(--red)' : (r.drift_severity_D > 0.3 ? 'var(--orange)' : 'var(--teal)');

    // SHAP bars
    let shapBars = '';
    const maxAbsShap = Math.max(...r.shap_explanation.map(s => Math.abs(s.value)), 0.01);
    r.shap_explanation.forEach(s => {
        const pct = Math.abs(s.value) / maxAbsShap * 100;
        const color = s.value > 0 ? 'var(--red)' : 'var(--blue)';
        const dir = s.value > 0 ? '→ Churn' : '→ Retain';
        shapBars += `<div class="shap-bar-container">
            <div class="shap-feature" title="${s.feature}">${s.feature} = ${s.feature_value}</div>
            <div class="shap-bar-track">
                <div class="shap-bar-fill" style="width:${pct}%; background:${color};"></div>
            </div>
            <div class="shap-val" style="color:${color};">${s.value > 0 ? '+' : ''}${s.value.toFixed(3)}</div>
        </div>`;
    });

    const verdictClass = r.would_defer ? 'verdict-defer' : 'verdict-auto';
    const verdictIcon = r.would_defer ? '⚠️' : '✅';
    const verdictTitle = r.would_defer ? 'DEFER TO ANALYST' : 'AUTO-PROCESS';
    const verdictSub = r.would_defer
        ? 'High composite score — this case needs human review'
        : 'Low risk — model is confident enough to auto-process';

    el.innerHTML = `
    <div class="grid-3" style="margin-bottom:20px;">
        <div class="card" style="text-align:center;">
            <div class="gauge-container">
                <div class="gauge-ring" style="background: conic-gradient(${gaugeColor} ${probPct}%, var(--bg2) 0);">
                    <div class="gauge-inner">
                        <div class="gauge-value" style="color:${gaugeColor};">${probPct}%</div>
                        <div class="gauge-label">Churn Prob</div>
                    </div>
                </div>
                <div class="gauge-caption">Prediction: <span class="badge ${r.prediction === 1 ? 'badge-red' : 'badge-green'}">${r.prediction_label}</span></div>
            </div>
        </div>
        <div class="card" style="text-align:center;">
            <div class="gauge-container">
                <div class="gauge-ring" style="background: conic-gradient(${driftColor} ${(r.drift_severity_D*100).toFixed(0)}%, var(--bg2) 0);">
                    <div class="gauge-inner">
                        <div class="gauge-value" style="color:${driftColor};">${(r.drift_severity_D*100).toFixed(0)}%</div>
                        <div class="gauge-label">Drift D(x)</div>
                    </div>
                </div>
                <div class="gauge-caption">Global normalized severity</div>
            </div>
        </div>
        <div class="card" style="text-align:center;">
            <div class="gauge-container">
                <div class="gauge-ring" style="background: conic-gradient(var(--purple) ${Math.min(100,(r.composite_score/1.5*100)).toFixed(0)}%, var(--bg2) 0);">
                    <div class="gauge-inner">
                        <div class="gauge-value" style="color:var(--purple);">${r.composite_score.toFixed(3)}</div>
                        <div class="gauge-label">Score(x)</div>
                    </div>
                </div>
                <div class="gauge-caption">U + D + ${r.lambda}·U·D</div>
            </div>
        </div>
    </div>

    <div class="verdict ${verdictClass}">
        <div class="verdict-icon">${verdictIcon}</div>
        <div class="verdict-title">${verdictTitle}</div>
        <div class="verdict-sub">${verdictSub}</div>
    </div>

    <div class="card" style="margin-top:20px;">
        <div class="card-title">🧬 SHAP Feature Attribution (Top 10)</div>
        <p style="font-size:11px; color:var(--text3); margin-bottom:12px;">Red bars push toward churn, blue bars push toward retention. Base value: ${r.shap_base_value}</p>
        ${shapBars}
    </div>

    <div class="card">
        <div class="card-title">📋 Score Breakdown</div>
        <table class="data-tbl" style="max-width:500px;">
            <tr><td>U(x) — Calibrated uncertainty</td><td style="font-weight:700;">${r.uncertainty_U.toFixed(4)}</td></tr>
            <tr><td>D(x) — Global drift severity</td><td style="font-weight:700;">${r.drift_severity_D.toFixed(4)}</td></tr>
            <tr><td>λ (interaction weight)</td><td style="font-weight:700;">${r.lambda}</td></tr>
            <tr><td>λ · U · D (interaction term)</td><td style="font-weight:700;">${(r.lambda * r.uncertainty_U * r.drift_severity_D).toFixed(4)}</td></tr>
            <tr style="font-size:14px;"><td><strong>Score(x) = U + D + λ·U·D</strong></td><td style="font-weight:800; color:var(--purple);">${r.composite_score.toFixed(4)}</td></tr>
            <tr><td>Decision threshold</td><td>${r.threshold}</td></tr>
        </table>
    </div>`;
}

// ── Accuracy ──
async function loadAccuracy() {
    const el = document.getElementById('accuracy-content');
    if (el.dataset.loaded) return;

    const resp = await fetch('/api/accuracy');
    const a = await resp.json();
    const o = a.overall;
    const cm = o.confusion_matrix;

    let windowRows = '';
    a.per_window.forEach(w => {
        windowRows += `<tr>
            <td><strong>W${w.window_id}</strong></td>
            <td>${w.n_customers}</td>
            <td>${w.churn_rate}%</td>
            <td>${w.accuracy}</td>
            <td>${w.precision}</td>
            <td>${w.recall}</td>
            <td><span class="badge ${w.f1 > 0.25 ? 'badge-green' : 'badge-orange'}">${w.f1}</span></td>
            <td>${w.auc}</td>
        </tr>`;
    });

    el.innerHTML = `
    <div class="kpi-row">
        <div class="kpi"><div class="kpi-label">Overall Accuracy</div><div class="kpi-value" style="color:var(--blue);">${(o.accuracy*100).toFixed(1)}%</div><div class="kpi-sub">${o.n_total} customers (W1-W9)</div></div>
        <div class="kpi"><div class="kpi-label">Precision</div><div class="kpi-value" style="color:var(--teal);">${(o.precision*100).toFixed(1)}%</div><div class="kpi-sub">Of predicted churners</div></div>
        <div class="kpi"><div class="kpi-label">Recall</div><div class="kpi-value" style="color:var(--orange);">${(o.recall*100).toFixed(1)}%</div><div class="kpi-sub">Of actual churners caught</div></div>
        <div class="kpi"><div class="kpi-label">F1 Score</div><div class="kpi-value" style="color:var(--purple);">${o.f1}</div><div class="kpi-sub">Harmonic mean P/R</div></div>
        <div class="kpi"><div class="kpi-label">AUC-ROC</div><div class="kpi-value" style="color:var(--green);">${o.auc}</div><div class="kpi-sub">Discrimination ability</div></div>
    </div>

    <div class="grid-2">
        <div class="card">
            <div class="card-title">🔢 Confusion Matrix (All Windows)</div>
            <div class="cm-grid">
                <div class="cm-corner"></div>
                <div class="cm-label">Pred 0</div>
                <div class="cm-label">Pred 1</div>
                <div class="cm-label" style="writing-mode:vertical-lr; transform:rotate(180deg);">Actual 0</div>
                <div class="cm-cell cm-tn">${cm[0][0]}<div class="cm-sub">True Neg</div></div>
                <div class="cm-cell cm-fp">${cm[0][1]}<div class="cm-sub">False Pos</div></div>
                <div class="cm-label" style="writing-mode:vertical-lr; transform:rotate(180deg);">Actual 1</div>
                <div class="cm-cell cm-fn">${cm[1][0]}<div class="cm-sub">False Neg</div></div>
                <div class="cm-cell cm-tp">${cm[1][1]}<div class="cm-sub">True Pos</div></div>
            </div>
            <p style="text-align:center; font-size:11px; color:var(--text3); margin-top:12px;">
                Threshold = ${o.threshold} (Platt-calibrated F1-optimal)
            </p>
        </div>
        <div class="card">
            <div class="card-title">📈 ROC Curve (AUC = ${o.auc})</div>
            <div style="height:260px; background:var(--bg2); border-radius:8px; position:relative; overflow:hidden; padding:20px;">
                <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" style="position:absolute; top:0; left:0;">
                    <line x1="0" y1="0" x2="100" y2="100" stroke="var(--text3)" stroke-width="0.5" stroke-dasharray="2"/>
                    <polyline fill="none" stroke="var(--blue)" stroke-width="1.5"
                        points="${a.roc_curve.map(p => `${p.fpr*100},${(1-p.tpr)*100}`).join(' ')}"/>
                    <polyline fill="rgba(59,130,246,0.1)" stroke="none"
                        points="0,100 ${a.roc_curve.map(p => `${p.fpr*100},${(1-p.tpr)*100}`).join(' ')} 100,100"/>
                </svg>
                <div style="position:absolute;bottom:8px;right:16px;font-size:10px;color:var(--text3);">FPR →</div>
                <div style="position:absolute;top:8px;left:8px;font-size:10px;color:var(--text3);">↑ TPR</div>
            </div>
        </div>
    </div>

    <div class="card">
        <div class="card-title">📊 Per-Window Model Performance</div>
        <div style="overflow-x:auto;">
            <table class="data-tbl">
                <thead><tr><th>Window</th><th>N</th><th>Churn %</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr></thead>
                <tbody>${windowRows}</tbody>
            </table>
        </div>
    </div>

    <div class="card">
        <div class="card-title">🎯 Calibration Check</div>
        <p style="font-size:11px; color:var(--text3); margin-bottom:12px;">Predicted probability buckets vs. actual churn rate</p>
        <div style="overflow-x:auto;">
            <table class="data-tbl" style="max-width:500px;">
                <thead><tr><th>Predicted Bucket</th><th>Avg Predicted</th><th>Actual Churn Rate</th><th>Count</th></tr></thead>
                <tbody>${a.calibration.map(c => `<tr><td>${(c.predicted*100).toFixed(0)}%</td><td>${(c.predicted*100).toFixed(1)}%</td><td>${(c.actual*100).toFixed(1)}%</td><td>${c.count}</td></tr>`).join('')}</tbody>
            </table>
        </div>
    </div>`;

    el.dataset.loaded = 'true';
}

// ── Explorer ──
async function loadExplorerPage(page) {
    const win = document.getElementById('exp-window').value;
    const filter = document.getElementById('exp-filter').value;
    const sort = document.getElementById('exp-sort').value;

    const resp = await fetch(`/api/customers?page=${page}&per_page=30&window=${win}&filter=${filter}&sort=${sort}&dir=desc`);
    const data = await resp.json();

    let rows = '';
    data.data.forEach(c => {
        const errBadge = c.is_model_error ? '<span class="badge badge-red">Error</span>' : '<span class="badge badge-green">OK</span>';
        const defBadge = c.is_deferred ? '<span class="badge badge-orange">Deferred</span>' : '';
        const predBadge = c.model_prediction === 1 ? '<span class="badge badge-red">Churn</span>' : '<span class="badge badge-blue">Retain</span>';
        rows += `<tr onclick="loadCustomerDetail(${c.customer_index})" style="cursor:pointer;">
            <td><strong>#${c.customer_index}</strong></td>
            <td>W${c.window_id}</td>
            <td>${predBadge}</td>
            <td>${c.actual_churn}</td>
            <td>${errBadge}</td>
            <td>${(c.predicted_churn_prob*100).toFixed(1)}%</td>
            <td>${c.drift_severity_D.toFixed(3)}</td>
            <td>${c.composite_score.toFixed(3)}</td>
            <td>${defBadge}</td>
        </tr>`;
    });

    document.getElementById('explorer-table').innerHTML = `
        <table class="data-tbl">
            <thead><tr><th>ID</th><th>Win</th><th>Prediction</th><th>Actual</th><th>Status</th><th>P(Churn)</th><th>D(x)</th><th>Score</th><th>Deferred</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>`;

    // Pagination
    let pagHtml = `<span>Page ${data.page} of ${data.pages} (${data.total} total)</span>`;
    if (data.page > 1) pagHtml += `<button onclick="loadExplorerPage(${data.page - 1})">← Prev</button>`;
    for (let p = Math.max(1, data.page - 2); p <= Math.min(data.pages, data.page + 2); p++) {
        pagHtml += `<button class="${p === data.page ? 'active' : ''}" onclick="loadExplorerPage(${p})">${p}</button>`;
    }
    if (data.page < data.pages) pagHtml += `<button onclick="loadExplorerPage(${data.page + 1})">Next →</button>`;
    document.getElementById('explorer-pagination').innerHTML = pagHtml;
}

async function loadCustomerDetail(id) {
    const resp = await fetch(`/api/customer/${id}`);
    const d = await resp.json();
    if (d.error) return;

    const el = document.getElementById('customer-detail');
    el.style.display = 'block';
    el.scrollIntoView({ behavior: 'smooth' });

    const maxAbsShap = Math.max(...d.shap_explanation.map(s => Math.abs(s.value)), 0.01);
    let shapBars = '';
    d.shap_explanation.forEach(s => {
        const pct = Math.abs(s.value) / maxAbsShap * 100;
        const color = s.value > 0 ? 'var(--red)' : 'var(--blue)';
        shapBars += `<div class="shap-bar-container">
            <div class="shap-feature">${s.feature} = ${s.feature_value}</div>
            <div class="shap-bar-track"><div class="shap-bar-fill" style="width:${pct}%; background:${color};"></div></div>
            <div class="shap-val" style="color:${color};">${s.value > 0 ? '+' : ''}${s.value.toFixed(3)}</div>
        </div>`;
    });

    const errStatus = d.is_error ? '❌ Model Error' : '✅ Correct';

    el.innerHTML = `
        <div class="card-title">🔍 Customer #${d.customer_index} — Window ${d.window_id}</div>
        <div class="grid-3" style="margin-bottom:16px;">
            <div style="text-align:center;">
                <div style="font-size:10px;color:var(--text3);text-transform:uppercase;">P(Churn)</div>
                <div style="font-size:28px;font-weight:800;color:${d.churn_probability > 0.3 ? 'var(--red)' : 'var(--blue)'};">${(d.churn_probability*100).toFixed(1)}%</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:10px;color:var(--text3);text-transform:uppercase;">Drift D(x)</div>
                <div style="font-size:28px;font-weight:800;color:var(--teal);">${d.drift_severity_D.toFixed(3)}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:10px;color:var(--text3);text-transform:uppercase;">Score(x)</div>
                <div style="font-size:28px;font-weight:800;color:var(--purple);">${d.composite_score.toFixed(3)}</div>
            </div>
        </div>
        <div style="display:flex; gap:12px; margin-bottom:16px; flex-wrap:wrap;">
            <span class="badge ${d.prediction === 1 ? 'badge-red' : 'badge-green'}">Predicted: ${d.prediction === 1 ? 'CHURN' : 'RETAIN'}</span>
            <span class="badge ${d.actual_churn === 1 ? 'badge-red' : 'badge-green'}">Actual: ${d.actual_churn === 1 ? 'CHURNED' : 'STAYED'}</span>
            <span class="badge ${d.is_error ? 'badge-red' : 'badge-green'}">${errStatus}</span>
            ${d.is_deferred ? '<span class="badge badge-orange">DEFERRED</span>' : '<span class="badge badge-blue">AUTO-PROCESSED</span>'}
        </div>
        <div style="margin-top:12px;">
            <div style="font-weight:600; margin-bottom:8px;">SHAP Feature Attribution</div>
            ${shapBars}
        </div>
        <button class="btn btn-secondary" style="margin-top:12px;" onclick="document.getElementById('customer-detail').style.display='none';">Close</button>
    `;
}

// Init
loadFeatures();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  L2D Interactive Model Testing UI")
    print("  Open: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
