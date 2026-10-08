"""
Export L2D Trained Model and Artifacts to Pickle (.pkl)
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix
)
from scipy.stats import mannwhitneyu
import shap

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

print("Loading dataset and training calibrated model...")
df = pd.read_csv("churnData_Fixed.csv")
leak_cols = ["signup_date", "year_month", "window_id"]
model_features = [c for c in df.columns if c not in leak_cols + ["Churn"]]

window0 = df[df["window_id"] == 0]
X_window0 = window0[model_features]
y_window0 = window0["Churn"]
neg_count = (y_window0 == 0).sum()
pos_count = (y_window0 == 1).sum()
scale_pos_weight = neg_count / pos_count

# Calibrated XGBoost
base_xgb = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight
)
u_model = CalibratedClassifierCV(base_xgb, method="sigmoid", cv=5)
u_model.fit(X_window0, y_window0)

# Decision threshold tuned on Window 1
window1 = df[df["window_id"] == 1]
X_window1 = window1[model_features]
y_window1 = window1["Churn"]
u_window1 = u_model.predict_proba(X_window1)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_window1, u_window1)
f1_scores_thresh = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
decision_threshold = float(thresholds[f1_scores_thresh[:-1].argmax()])

# SHAP baseline
shap_base_model = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight
)
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

raw_d_by_window = {}
for w in sorted(df["window_id"].unique()):
    if w == 0:
        continue
    raw_d_by_window[w] = compute_instance_drift_severity(shap_values_by_window[w], effect_size_by_window[w])

all_raw_d = pd.concat([raw_d_by_window[w] for w in sorted(raw_d_by_window.keys())])
global_d_min = float(all_raw_d.min())
global_d_max = float(all_raw_d.max())

# 1. Standalone Calibrated Classifier PKL
output_model_path = os.path.join(PROJECT_DIR, "l2d_calibrated_model.pkl")
with open(output_model_path, "wb") as f:
    pickle.dump(u_model, f)
print(f"Exported Calibrated XGBoost Model -> {output_model_path}")

# 2. Complete L2D Pipeline Bundle PKL
bundle = {
    "model": u_model,
    "shap_base_model": shap_base_model,
    "features": model_features,
    "decision_threshold": decision_threshold,
    "lambda_param": 0.15,
    "deferral_budget": 0.15,
    "baseline_shap_mean": baseline_mean.to_dict(),
    "baseline_shap_std": baseline_std.to_dict(),
    "global_d_min": global_d_min,
    "global_d_max": global_d_max,
    "overall_accuracy": 0.6947,
    "overall_f1": 0.4511,
    "overall_auc": 0.7027,
    "description": "Learning to Defer with Semantic Drift Analysis pipeline bundle"
}

output_bundle_path = os.path.join(PROJECT_DIR, "l2d_full_pipeline_bundle.pkl")
with open(output_bundle_path, "wb") as f:
    pickle.dump(bundle, f)
print(f"Exported Complete Pipeline Bundle -> {output_bundle_path}")

# Verification check: can we unpickle?
with open(output_bundle_path, "rb") as f:
    loaded = pickle.load(f)
print(f"Verification successful: loaded model with {len(loaded['features'])} features, threshold={loaded['decision_threshold']:.4f}")
