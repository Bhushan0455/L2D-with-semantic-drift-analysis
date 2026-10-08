"""
Machine Learning models and calibration routines
"""

from typing import Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve

from src.config import XGB_PARAMS, CALIBRATION_METHOD, CALIBRATION_CV


def train_calibrated_model(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[CalibratedClassifierCV, XGBClassifier]:
    """
    Trains base XGBoost with scale_pos_weight and applies Platt Scaling
    using 5-fold cross validation. Also returns the fitted base model for SHAP TreeExplainer.
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)

    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    base_model = XGBClassifier(**params)
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method=CALIBRATION_METHOD,
        cv=CALIBRATION_CV
    )
    calibrated_model.fit(X_train, y_train)

    # Fit standalone base model for TreeExplainer
    shap_base_model = XGBClassifier(**params)
    shap_base_model.fit(X_train, y_train)

    return calibrated_model, shap_base_model


def optimize_decision_threshold(model: CalibratedClassifierCV, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Tuning the decision threshold on Window 1 to maximize F1 score.
    """
    probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = f1_scores[:-1].argmax()
    return float(thresholds[best_idx])
