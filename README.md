# Learning to Defer (L2D) with Semantic Data Drift Analysis (SDDA)

> **Semester V ML Mini-Project**  
> Telecom Customer Churn Prediction with Cost-Aware Deferral & SHAP-based Drift Detection

---

## 📌 Project Overview

Production machine learning models frequently degrade over time due to **distribution shifts** and **semantic data drift** (e.g., changes in customer behavior, competitor pricing, or macroeconomic inflation). 

This project implements a **Learning to Defer (L2D)** framework for telecom churn prediction:
1. **Calibrated Churn Model:** XGBoost with Platt Scaling (sigmoid calibration) providing well-calibrated uncertainty scores $U(x)$.
2. **Semantic Drift Detection (SDDA):** SHAP-based instance-level feature attributions, paired with Mann-Whitney U hypothesis testing and rank-biserial effect sizes to compute per-customer drift severity $D(x)$.
3. **Cost-Aware Deferral Policy:** A composite interaction score $Score(x) = U(x) + D(x) + \lambda \cdot U(x) \cdot D(x)$ that dynamically routes high-risk, ambiguous, or drifted customer cases to human intervention under a strict budget constraint.
4. **Interactive Testing UI & Power BI Integration:** A Flask-based web application with live SHAP waterfall attributions, alongside export pipelines for business intelligence reporting.

---

## 📊 Model Performance Highlights

Across **6,338 deployment samples** spanning 9 sequential time windows:

| Metric | Overall Value | Window 1 (Baseline) | Window 9 (Severe Drift) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **69.47%** | **79.12%** | **54.33%** |
| **ROC-AUC** | **0.7027** | **0.7456** | **0.7068** |
| **Precision** | **47.18%** | 24.05% | 80.92% |
| **Recall** | **43.21%** | 58.46% | 32.63% |
| **F1 Score** | **0.4511** | 0.3408 | 0.4651 |
| **Churn Rate**| 29.0% | 9.2% | 60.9% |

> **Key Finding:** As distribution drift escalates (churn rate jumps from 9.2% to 60.9%), the static model's accuracy drops from **~79%** to **~54%**. The Learning to Defer policy prevents costly automated misclassifications by proactively deferring ambiguous cases.

---

## 🗂️ Industry-Grade Project Architecture

```text
├── data/
│   ├── raw/                       # Original raw telecom records (telco.xls)
│   ├── processed/                 # Cleaned dataset with time windows & CPI (churnData_Fixed.csv)
│   └── outputs/                   # Power BI feeds & Phase CSV/Excel results
├── models/                        # Serialized production model artifacts
│   ├── l2d_calibrated_model.pkl   # Platt-calibrated XGBoost classifier
│   └── l2d_full_pipeline_bundle.pkl # End-to-end bundle (model, SHAP, scaler, thresholds)
├── src/                           # Modular production ML package
│   ├── config.py                  # Central configuration & path management
│   ├── data_loader.py             # Data loading & feature inspection
│   ├── models.py                  # Calibrated XGBoost & threshold optimizer
│   ├── drift.py                   # SHAP TreeExplainer & Mann-Whitney U drift detector
│   ├── deferral.py                # Composite scoring & budget deferral routing
│   └── evaluation.py              # Classification metrics & confusion matrices
├── app/                           # Interactive Flask web application
│   └── app.py                     # Web server & API endpoints
├── dashboards/                    # Business Intelligence assets & visualizations
│   ├── Phase6_dashboard_sketch.html # Standalone interactive dashboard
│   └── build_dashboard.py         # Dashboard build automation
├── notebooks/                     # Academic & experimental notebooks (Phase 1–5)
├── scripts/                       # Automated CLI execution scripts
│   ├── export_models.py           # Model serialization CLI
│   ├── run_full_sweep.py          # Phase 5b ablation & sweep CLI
│   └── export_powerbi_data.py     # Power BI data generation CLI
├── tests/                         # Unit & integration tests
│   └── test_pipeline.py           # Pipeline sanity checks
├── docs/                          # Project documentation & explainer PDFs
├── run_interactive_ui.bat         # 1-click Windows batch launcher for UI
├── requirements.txt               # Required Python packages
└── README.md                      # Comprehensive project guide
```

---

## ⚙️ Prerequisites & Setup

### 1. Requirements
* **Operating System:** Windows, macOS, or Linux
* **Python Version:** Python 3.10, 3.11, or 3.12 (Anaconda recommended)

### 2. Environment Setup

#### Option A: Using Anaconda / Conda (Recommended)
```bash
# Create and activate a conda environment
conda create -n l2d python=3.11 -y
conda activate l2d

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using Standard Python Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Interactive Testing Web UI (Easiest & Recommended)

Launch the interactive web application to test model predictions live, inspect SHAP waterfalls, view accuracy metrics, and explore customer deferral records:

* **Windows 1-Click:** Double-click [`run_interactive_ui.bat`](run_interactive_ui.bat)  
* **Command Line:**
  ```bash
  python Phase6_interactive_app.py
  ```
* Open your browser at: **`http://localhost:5000`**

#### Web App Features:
* **Predict Tab:** Enter customer attributes (tenure, contract, charges, CPI inflation) or click sample presets to view:
  * Predicted Churn Probability $P(\text{Churn})$
  * Uncertainty Score $U(x)$
  * Drift Severity $D(x)$
  * Composite Score $Score(x)$
  * Automated vs. Deferred Routing decision
  * Interactive **SHAP Waterfall Attribution bars** showing which features increased or decreased churn risk.
* **Accuracy Tab:** Interactive confusion matrix, ROC curve, and per-window accuracy degradation breakdown.
* **Customer Explorer Tab:** Browse 6,338 historical predictions, filter by deferred cases or model errors, and inspect detailed feature contributions.

---

### 2. Using the Trained Model (`.pkl`)

The project outputs two pre-trained pickle files:
1. `l2d_calibrated_model.pkl`: The calibrated classifier (`CalibratedClassifierCV`) directly ready for standard prediction.
2. `l2d_full_pipeline_bundle.pkl`: Complete pipeline bundle containing the model, 32 feature names, tuned decision threshold ($\tau = 0.1298$), $\lambda = 0.15$, deferral budget ($15\%$), and baseline SHAP reference vectors.

#### Python Example: Inference with `.pkl`
```python
import pickle
import pandas as pd

# Load the full pipeline bundle
with open("l2d_full_pipeline_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
features = bundle["features"]
threshold = bundle["decision_threshold"]  # 0.1298

# Sample customer DataFrame with required 32 features
sample_df = pd.DataFrame([{
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "tenure": 12,
    "PhoneService": 1,
    "PaperlessBilling": 1,
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0,
    "india_cpi": 142.5,
    "cpi_mom_pct_change": 0.45,
    # ... remaining one-hot encoded categorical features
}])

# Predict calibrated churn probability
churn_prob = model.predict_proba(sample_df[features])[:, 1][0]
is_churn = int(churn_prob >= threshold)

print(f"P(Churn): {churn_prob:.4f}")
print(f"Prediction: {'CHURN' if is_churn == 1 else 'RETAIN'}")
```

To re-train and export updated `.pkl` files at any time, run:
```bash
python export_model_pkl.py
```

---

### 3. Power BI Export Pipeline

To generate fresh CSV feeds for Power BI reporting:
```bash
python Phase6_powerbi_export.py
```

This generates:
* `phase6_customer_deferral_data.csv`: Complete customer-level predictions, scores, and SHAP top-3 drivers.
* `phase6_deferred_queue.csv`: Prioritized queue of customers deferred for human specialist review.
* `phase6_drift_summary.csv`: Window-by-window drift summary with percentage of drifting features.
* `phase6_drift_detail.csv`: Per-feature Mann-Whitney U statistics and effect sizes.
* `phase6_feature_drift_trend.csv`: Temporal drift trajectories across deployment windows.

You can also double-click [`Phase6_dashboard_sketch.html`](Phase6_dashboard_sketch.html) to open an offline executive dashboard in your browser.

---

### 4. Running the Jupyter Notebooks

If you want to step through the research and development phases:
```bash
jupyter notebook
```
Execute in chronological order:
1. `Phase1.ipynb` — EDA, data cleaning, and macro CPI merging.
2. `Phase2.ipynb` — XGBoost model training and Platt scaling calibration.
3. `Phase3.ipynb` — SHAP values calculation and Mann-Whitney U drift testing.
4. `Phase4.ipynb` — Grid search for interaction parameter $\lambda$ and budget evaluation.
5. `Phase5.ipynb` — Ablation comparisons and risk-coverage curves.

---

### 5. Running Automated Tests

To run the automated pipeline test suite:
```bash
python -m unittest discover tests
```

---

## 🛠️ Troubleshooting & FAQs

### Q1: `ImportError: numpy: No module named 'numpy'` or Python version mismatch
* **Cause:** Multiple Python installations exist on your machine (e.g., Python 3.13 default without libraries vs. Anaconda Python).
* **Fix:** Ensure you run commands using your configured virtual environment or Anaconda Python executable:
  ```bash
  # Activate conda first:
  conda activate l2d
  # Or explicitly use Anaconda python:
  C:\Users\<username>\anaconda3\python.exe Phase6_interactive_app.py
  ```

### Q2: Port 5000 is already in use
* If port 5000 is occupied by another service, change `port=5000` to `port=5050` at the bottom of [`Phase6_interactive_app.py`](Phase6_interactive_app.py) line 1284.

### Q3: Missing dataset file
* Ensure `churnData_Fixed.csv` is present in the project directory. The pipeline automatically checks and loads this file relative to the script's directory.

---

## 👥 Contributors & Academic Context
* **Course:** Semester V Machine Learning Mini-Project
* **Topic:** Learning to Defer (L2D) with Semantic Data Drift Analysis (SDDA)
* **Dataset:** IBM Telco Churn merged with macroeconomic monthly CPI series.
