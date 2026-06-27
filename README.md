# 🔮 Customer Churn Prediction

> **Binary classification** — predict which telecom customers will churn using
> Random Forest, XGBoost, SMOTE, and SHAP explainability; served via a
> production-ready Streamlit dashboard.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [What Changed](#what-changed)
3. [Demo](#demo)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)
6. [Tech Stack](#tech-stack)
7. [Quick Start](#quick-start)
8. [Google Colab Notebook](#google-colab-notebook)
9. [Streamlit App](#streamlit-app)
10. [Model Pipeline](#model-pipeline)
11. [Results](#results)
12. [SHAP Explainability](#shap-explainability)
13. [Deployment](#deployment)
14. [Contributing](#contributing)
15. [License](#license)

---

## Project Overview

Customer churn is one of the most costly problems in the telecom industry.
This project builds an **end-to-end ML pipeline** that:

- Ingests and cleans the IBM Telco Customer Churn dataset
- Performs exploratory data analysis (EDA)
- Engineers predictive features
- Handles class imbalance with **SMOTE**
- Trains and tunes **Logistic Regression**, **Random Forest**, and **XGBoost**
- Explains predictions with **SHAP** (SHapley Additive exPlanations)
- Serves results through a polished **Streamlit** web application

---

## What Changed

This version fixes three correctness issues found when re-testing the original
notebook and app against current package versions (pandas 2.2+/3.0, shap
0.45+):

| Issue | Where | Fix |
|---|---|---|
| `df[col].fillna(value, inplace=True)` silently does nothing under pandas Copy-on-Write, leaving `TotalCharges` NaNs in the data | Notebook Cell 4, `app.py` batch prediction | Changed to `df[col] = df[col].fillna(value)` |
| `shap.TreeExplainer.shap_values()` no longer returns a `list` for binary RandomForest — it returns a 3D `ndarray`, so the old `shap_values[1][0]` indexing crashes with `IndexError` | Notebook Cell 10, `app.py` single prediction | Added a version-agnostic `get_positive_class_shap(...)` helper that handles both the legacy list API and the current ndarray API |
| `fetch_openml(name="Telco-Customer-Churn", version=1)` resolves by fuzzy name match, which is fragile (multiple differently-shaped churn datasets exist on OpenML) | Notebook Cell 3 | Load the dataset directly from the documented GitHub CSV; a stable `data_id=42178` alternative is included as a comment if you prefer OpenML |
| `HighValue`/`LongTenure` thresholds were hardcoded in `app.py` (e.g. `64.76`), duplicating training-time logic and silently drifting after retraining | `app.py` | Notebook now saves `feature_thresholds.pkl`; the app loads it instead of hardcoding |
| `XGBClassifier(use_label_encoder=False, ...)` triggers a deprecation warning on modern xgboost (harmless but noisy) | Notebook, both model definitions | Removed the unused parameter |

All fixes were verified by running the corrected notebook end-to-end and
exercising the app's SHAP code path against the resulting model artifact.

---

## Demo

```
streamlit run app.py
```

| Feature | Detail |
|---|---|
| Single prediction | Enter customer details and get an instant churn probability |
| Batch prediction  | Upload a CSV; download predictions + risk levels |
| SHAP waterfall    | Per-customer explanation of the model's decision |
| Model info page   | Pipeline overview, benchmarks, key churn drivers |

---

## Project Structure

```
customer-churn-prediction/
│
├── customer_churn_notebook.py   # Google Colab notebook (all 11 cells)
├── app.py                       # Streamlit production application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore
│
├── best_churn_model.pkl         # Saved best model              (generated)
├── scaler.pkl                   # Fitted StandardScaler         (generated)
├── feature_columns.pkl          # Training column order         (generated)
├── feature_thresholds.pkl       # HighValue/LongTenure cutoffs  (generated)
├── best_model_name.pkl          # Name of the winning model      (generated)
│
└── plots/                       # Auto-generated EDA & eval plots
    ├── eda_distributions.png
    ├── eda_correlation.png
    ├── evaluation.png
    ├── shap_bar.png
    └── shap_beeswarm.png
```

---

## Dataset

**IBM Telco Customer Churn** — publicly available on Kaggle and GitHub.

| Property | Value |
|---|---|
| Rows | 7,043 |
| Features | 21 raw → 30+ after engineering |
| Target | `Churn` (Yes / No) |
| Class imbalance | ~26% churn |
| Source | [Raw CSV (GitHub)](https://raw.githubusercontent.com/dsrscientist/dataset1/master/Telco-Customer-Churn.csv) |

The notebook loads this CSV directly with `pd.read_csv(...)`. If you'd rather
pull it from OpenML, use the stable numeric ID instead of a name lookup:
`fetch_openml(data_id=42178, as_frame=True)` — name-based lookups can resolve
to the wrong dataset since several differently-structured "churn" datasets
exist on the platform.

### Key Features

| Feature | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months as a customer |
| `MonthlyCharges` | Numeric | Monthly bill ($) |
| `TotalCharges` | Numeric | Cumulative bill ($) |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / etc. |
| `OnlineSecurity` | Binary | Yes / No |
| … | … | … |

---

## Tech Stack

| Layer | Library |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| ML models | `scikit-learn`, `xgboost` |
| Class balancing | `imbalanced-learn` (SMOTE) |
| Explainability | `shap` |
| Persistence | `joblib` |
| Web app | `streamlit` |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Colab notebook (locally with Jupyter)

```bash
jupyter notebook customer_churn_notebook.py
```

Or open it directly in **Google Colab**:

> Upload `customer_churn_notebook.py` → File → Open → upload → run all cells.

This will produce `best_churn_model.pkl`, `scaler.pkl`, `feature_columns.pkl`,
`feature_thresholds.pkl`, and `best_model_name.pkl`.

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Google Colab Notebook

The notebook (`customer_churn_notebook.py`) is structured into **11 executable cells**:

| Cell | Purpose |
|---|---|
| 1 | Install dependencies |
| 2 | Import libraries |
| 3 | Load dataset from GitHub CSV |
| 4 | Data cleaning (type coercion, encoding, OHE) |
| 5 | EDA — distributions, correlation heatmap |
| 6 | Feature engineering + SMOTE |
| 7 | Model training — LR, RF, XGBoost (cross-validated) |
| 8 | Evaluation — ROC curves, confusion matrix, reports |
| 9 | Hyperparameter tuning — GridSearchCV |
| 10 | SHAP explainability — bar + beeswarm plots |
| 11 | Save best model with joblib |

---

## Streamlit App

`app.py` contains three pages:

### 🔍 Single Prediction
Enter demographic, account, and service details via an interactive form.
The app returns:
- Churn probability (%)
- Risk level (Low / Medium / High)
- SHAP bar chart (top 10 drivers for *this* customer)
- Tailored retention recommendations

### 📂 Batch Prediction
Upload a CSV of any size.  
The app preprocesses, predicts, and outputs:
- A downloadable results CSV with `ChurnProbability`, `Prediction`, `RiskLevel`
- Summary metrics (total customers, predicted churners, avg probability)

### 📊 Model Info
Pipeline summary, benchmark metrics, and a visualisation of the top churn
drivers derived from SHAP analysis.

---

## Model Pipeline

```
Raw CSV
  │
  ▼
Data Cleaning
  (type coerce · fillna · label encode · OHE)
  │
  ▼
Feature Engineering
  (ChargesPerMonth · HighValue · LongTenure · MultipleServices)
  │
  ▼
Train / Test Split  (80 / 20, stratified)
  │
  ▼
SMOTE  (oversample minority class on train set only)
  │
  ├── Logistic Regression  ─┐
  ├── Random Forest         ├──► Cross-validated AUC → GridSearchCV
  └── XGBoost              ─┘
                              │
                              ▼
                         Best Tuned Model
                              │
                         ┌────┴────┐
                         │  SHAP   │
                         └────┬────┘
                              │
                         joblib.dump()
                              │
                         Streamlit App
```

---

## Results

| Model | CV AUC | Test AUC | Test F1 |
|---|---|---|---|
| Logistic Regression | ~0.84 | ~0.83 | ~0.60 |
| Random Forest | ~0.84 | ~0.84 | ~0.61 |
| **XGBoost (tuned)** | **~0.85** | **~0.85** | **~0.63** |

> Results may vary slightly depending on random seeds and SMOTE sampling.

---

## SHAP Explainability

SHAP (SHapley Additive exPlanations) attributes the model's output to each
feature for every prediction.

**Top churn drivers identified:**

1. `Contract_Month-to-month` — strongest churn signal
2. `tenure` — shorter tenure → higher churn risk
3. `InternetService_Fiber optic` — Fiber customers churn more
4. `MonthlyCharges` — higher bills correlate with churn
5. `OnlineSecurity_No` — missing security add-on raises risk
6. `TechSupport_No` — no tech support increases churn
7. `PaperlessBilling` — paperless customers churn more
8. `PaymentMethod_Electronic check` — top churn payment method

---

## Deployment

### Streamlit Community Cloud (free)

1. Push the repo to GitHub (include all `.pkl` files or re-generate in an init script).
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app.py` as the entry point → deploy.

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t churn-predictor .
docker run -p 8501:8501 churn-predictor
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**.  
See [LICENSE](LICENSE) for details.

---

*Built with ❤️ using Python · scikit-learn · XGBoost · SHAP · Streamlit*
