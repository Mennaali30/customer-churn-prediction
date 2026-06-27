# app.py — Customer Churn Prediction · Streamlit Production App
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import os
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        text-align: center;
        border-left: 4px solid #4f8ef7;
    }
    .risk-high   { color: #d32f2f; font-size: 1.8rem; font-weight: 700; }
    .risk-medium { color: #f57c00; font-size: 1.8rem; font-weight: 700; }
    .risk-low    { color: #388e3c; font-size: 1.8rem; font-weight: 700; }
    .stButton>button {
        background-color: #4f8ef7;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        border: none;
    }
    .stButton>button:hover { background-color: #2563eb; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_ARTEFACTS = ["best_churn_model.pkl", "scaler.pkl", "feature_columns.pkl"]
# These two are optional for backward compatibility with older notebook runs,
# but recommended — see fallbacks below.
OPTIONAL_ARTEFACTS = ["feature_thresholds.pkl", "best_model_name.pkl"]


@st.cache_resource
def load_artefacts():
    model = joblib.load("best_churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("feature_columns.pkl")

    if os.path.exists("feature_thresholds.pkl"):
        thresholds = joblib.load("feature_thresholds.pkl")
    else:
        # Fallback for older artefacts that didn't save thresholds. This is
        # an approximation — re-run the notebook to get the exact value
        # used during training.
        thresholds = {"high_value_threshold": 64.76, "long_tenure_threshold": 24}

    if os.path.exists("best_model_name.pkl"):
        model_name = joblib.load("best_model_name.pkl")
    else:
        model_name = type(model).__name__

    return model, scaler, columns, thresholds, model_name


ARTEFACTS_EXIST = all(os.path.exists(f) for f in REQUIRED_ARTEFACTS)

if ARTEFACTS_EXIST:
    model, scaler, feature_columns, thresholds, best_model_name = load_artefacts()
    HIGH_VALUE_THRESHOLD = thresholds["high_value_threshold"]
    LONG_TENURE_THRESHOLD = thresholds["long_tenure_threshold"]
else:
    st.warning(
        "⚠️ Model artefacts not found. "
        "Please run the Colab notebook first to generate "
        "`best_churn_model.pkl`, `scaler.pkl`, and `feature_columns.pkl`."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🔮 Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Predict whether a telecom customer will churn '
    '— powered by XGBoost / Random Forest + SHAP explainability</p>',
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/predictive-chart.png", width=80)
    st.markdown("## Navigation")
    page = st.radio("", ["🔍 Single Prediction", "📂 Batch Prediction", "📊 Model Info"])
    st.divider()
    st.markdown(f"**Model:** {best_model_name}")
    st.markdown("**Dataset:** IBM Telco Customer Churn")
    st.markdown("**Features:** 30+")

# ─────────────────────────────────────────────────────────────────────────────
# Helper — SHAP value extraction (version-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
def get_positive_class_shap_row(shap_values) -> np.ndarray:
    """Normalize SHAP output to a 1D array of per-feature SHAP values for the
    positive (churn) class, for a single input row.

    Handles both the legacy shap API (list of per-class arrays) and the
    current shap API (single ndarray, possibly 3D with a trailing class
    axis for tree ensembles like RandomForest).
    """
    if isinstance(shap_values, list):
        # Legacy API: [class0_array, class1_array], each (n_samples, n_features)
        return np.asarray(shap_values[1])[0]

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # Current API for some tree models: (n_samples, n_features, n_classes)
        return arr[0, :, 1]
    # Already 2D: (n_samples, n_features) — typical for XGBoost binary clf
    return arr[0]


# ─────────────────────────────────────────────────────────────────────────────
# Helper — build input row
# ─────────────────────────────────────────────────────────────────────────────
def build_input_row(inputs: dict) -> pd.DataFrame:
    """Convert sidebar inputs into a one-row DataFrame matching training columns."""

    # ── numeric ──────────────────────────────────────────────────────────────
    row = {
        "tenure":          inputs["tenure"],
        "MonthlyCharges":  inputs["MonthlyCharges"],
        "TotalCharges":    inputs["TotalCharges"],
        # binary
        "Partner":         1 if inputs["Partner"]          == "Yes" else 0,
        "Dependents":      1 if inputs["Dependents"]       == "Yes" else 0,
        "PhoneService":    1 if inputs["PhoneService"]     == "Yes" else 0,
        "PaperlessBilling":1 if inputs["PaperlessBilling"] == "Yes" else 0,
        "MultipleLines":   1 if inputs["MultipleLines"]    == "Yes" else 0,
        "OnlineSecurity":  1 if inputs["OnlineSecurity"]   == "Yes" else 0,
        "OnlineBackup":    1 if inputs["OnlineBackup"]     == "Yes" else 0,
        "DeviceProtection":1 if inputs["DeviceProtection"] == "Yes" else 0,
        "TechSupport":     1 if inputs["TechSupport"]      == "Yes" else 0,
        "StreamingTV":     1 if inputs["StreamingTV"]      == "Yes" else 0,
        "StreamingMovies": 1 if inputs["StreamingMovies"]  == "Yes" else 0,
        # engineered — uses the SAME thresholds saved during training,
        # rather than a hardcoded value that can drift after retraining.
        "ChargesPerMonth": inputs["TotalCharges"] / (inputs["tenure"] + 1),
        "HighValue":       1 if inputs["MonthlyCharges"] > HIGH_VALUE_THRESHOLD else 0,
        "LongTenure":      1 if inputs["tenure"] > LONG_TENURE_THRESHOLD else 0,
        "MultipleServices": sum([
            1 if inputs[s] == "Yes" else 0
            for s in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                      "TechSupport", "StreamingTV", "StreamingMovies"]
        ]),
        # gender OHE
        "gender_Female": 1 if inputs["gender"] == "Female" else 0,
        "gender_Male":   1 if inputs["gender"] == "Male"   else 0,
        # InternetService OHE
        "InternetService_DSL":             1 if inputs["InternetService"] == "DSL"             else 0,
        "InternetService_Fiber optic":     1 if inputs["InternetService"] == "Fiber optic"     else 0,
        "InternetService_No":              1 if inputs["InternetService"] == "No"               else 0,
        # Contract OHE
        "Contract_Month-to-month":         1 if inputs["Contract"] == "Month-to-month"         else 0,
        "Contract_One year":               1 if inputs["Contract"] == "One year"               else 0,
        "Contract_Two year":               1 if inputs["Contract"] == "Two year"               else 0,
        # PaymentMethod OHE
        "PaymentMethod_Bank transfer (automatic)":
            1 if inputs["PaymentMethod"] == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)":
            1 if inputs["PaymentMethod"] == "Credit card (automatic)"   else 0,
        "PaymentMethod_Electronic check":
            1 if inputs["PaymentMethod"] == "Electronic check"          else 0,
        "PaymentMethod_Mailed check":
            1 if inputs["PaymentMethod"] == "Mailed check"              else 0,
    }

    df_row = pd.DataFrame([row])

    # Align to training columns (add any missing as 0, drop extras)
    for col in feature_columns:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[feature_columns]

    # Scale numeric features
    num_features = ["tenure", "MonthlyCharges", "TotalCharges",
                    "ChargesPerMonth", "MultipleServices"]
    df_row[num_features] = scaler.transform(df_row[num_features])

    return df_row


def risk_label(prob: float):
    if prob >= 0.70:
        return "🔴 HIGH RISK", "risk-high"
    elif prob >= 0.40:
        return "🟡 MEDIUM RISK", "risk-medium"
    else:
        return "🟢 LOW RISK", "risk-low"


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Single Prediction
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔍 Single Prediction":
    st.subheader("📋 Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender      = st.selectbox("Gender",      ["Male", "Female"])
        partner     = st.selectbox("Partner",     ["Yes", "No"])
        dependents  = st.selectbox("Dependents",  ["Yes", "No"])

        st.markdown("**Account**")
        tenure      = st.slider("Tenure (months)", 0, 72, 12)
        contract    = st.selectbox("Contract Type",
                                   ["Month-to-month", "One year", "Two year"])
        payment     = st.selectbox("Payment Method", [
                                   "Electronic check", "Mailed check",
                                   "Bank transfer (automatic)",
                                   "Credit card (automatic)"])
        paperless   = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col2:
        st.markdown("**Charges**")
        monthly     = st.number_input("Monthly Charges ($)",  10.0, 120.0, 65.0, 0.5)
        total       = st.number_input("Total Charges ($)",     0.0, 9000.0,
                                      round(float(tenure * monthly), 2), 10.0)

        st.markdown("**Phone Services**")
        phone       = st.selectbox("Phone Service",    ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines",   ["Yes", "No"])

        st.markdown("**Internet**")
        internet    = st.selectbox("Internet Service",
                                   ["Fiber optic", "DSL", "No"])

    with col3:
        st.markdown("**Add-on Services**")
        online_sec  = st.selectbox("Online Security",   ["Yes", "No"])
        online_bak  = st.selectbox("Online Backup",     ["Yes", "No"])
        device_prot = st.selectbox("Device Protection", ["Yes", "No"])
        tech_sup    = st.selectbox("Tech Support",      ["Yes", "No"])
        stream_tv   = st.selectbox("Streaming TV",      ["Yes", "No"])
        stream_mov  = st.selectbox("Streaming Movies",  ["Yes", "No"])

    st.divider()
    predict_btn = st.button("🔮 Predict Churn Probability", use_container_width=True)

    if predict_btn:
        inputs = {
            "gender": gender, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "Contract": contract, "PaymentMethod": payment,
            "PaperlessBilling": paperless, "MonthlyCharges": monthly,
            "TotalCharges": total, "PhoneService": phone,
            "MultipleLines": multi_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_bak,
            "DeviceProtection": device_prot, "TechSupport": tech_sup,
            "StreamingTV": stream_tv, "StreamingMovies": stream_mov,
        }

        df_input = build_input_row(inputs)
        prob     = model.predict_proba(df_input)[0, 1]
        label, css = risk_label(prob)

        st.subheader("🎯 Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{prob*100:.1f}%")
        r2.metric("Retention Probability", f"{(1-prob)*100:.1f}%")
        r3.markdown(f'<p class="{css}">{label}</p>', unsafe_allow_html=True)

        # Progress bar
        st.progress(float(prob))

        # SHAP waterfall for this customer
        st.subheader("🔍 SHAP Explanation (why this prediction?)")
        try:
            if isinstance(model, xgb.XGBClassifier):
                explainer = shap.TreeExplainer(model)
                raw_shap = explainer.shap_values(df_input)
            elif hasattr(model, "estimators_"):  # tree ensembles, e.g. RandomForest
                explainer = shap.TreeExplainer(model)
                raw_shap = explainer.shap_values(df_input)
            else:
                explainer = shap.LinearExplainer(model, df_input)
                raw_shap = explainer.shap_values(df_input)

            shap_vals = get_positive_class_shap_row(raw_shap)

            # Top 10 features by |SHAP|
            feature_names = list(df_input.columns)
            shap_series_abs = pd.Series(shap_vals, index=feature_names).abs().nlargest(10)
            top_features = shap_series_abs.index.tolist()
            top_signed = pd.Series(shap_vals, index=feature_names)[top_features]

            fig, ax = plt.subplots(figsize=(9, 5))
            colors = ["#d32f2f" if v > 0 else "#388e3c" for v in top_signed.sort_values()]
            top_signed.sort_values().plot(kind="barh", ax=ax, color=colors)
            ax.set_title("Top 10 Features Driving This Prediction", fontsize=13)
            ax.set_xlabel("SHAP Value (red = pushes toward churn, green = retains)")
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

        # Recommendations
        st.subheader("💡 Retention Recommendations")
        if prob >= 0.70:
            st.error("**Immediate action required.** This customer is very likely to churn.")
            recs = [
                "📞 Proactively reach out with a personalised retention offer.",
                "🎁 Offer a discount or loyalty reward on their next bill.",
                "📦 Upgrade or bundle their current plan at a reduced rate.",
                "🔧 Assign a dedicated account manager.",
            ]
        elif prob >= 0.40:
            st.warning("**Monitor closely.** This customer shows moderate churn risk.")
            recs = [
                "📧 Send an engagement email with product highlights.",
                "📋 Survey the customer to identify pain points.",
                "🌐 Offer complementary add-on services (Security, Backup).",
            ]
        else:
            st.success("**Low risk.** This customer is likely to stay.")
            recs = [
                "🤝 Keep delivering excellent service.",
                "🆙 Consider an upsell opportunity for premium features.",
            ]
        for r in recs:
            st.markdown(f"- {r}")


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Batch Prediction
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📂 Batch Prediction":
    st.subheader("📂 Batch Customer Churn Prediction")
    st.markdown(
        "Upload a CSV file containing customer records. "
        "The file must include the same columns used during training "
        "(tenure, MonthlyCharges, TotalCharges, Contract, etc.)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        raw = pd.read_csv(uploaded)
        st.markdown(f"**Rows uploaded:** {len(raw)}")
        st.dataframe(raw.head(), use_container_width=True)

        if st.button("🔮 Run Batch Prediction"):
            try:
                # ── minimal preprocessing identical to notebook ──────────────
                df_b = raw.copy()

                if "customerID" in df_b.columns:
                    cust_ids = df_b["customerID"].values
                    df_b.drop(columns=["customerID"], inplace=True)
                else:
                    cust_ids = np.arange(len(df_b))

                if "Churn" in df_b.columns:
                    df_b.drop(columns=["Churn"], inplace=True)

                df_b["TotalCharges"] = pd.to_numeric(df_b["TotalCharges"], errors="coerce")
                # Assign back instead of inplace=True on a column slice —
                # the latter can silently no-op under pandas Copy-on-Write.
                df_b["TotalCharges"] = df_b["TotalCharges"].fillna(df_b["TotalCharges"].median())

                binary_cols = [
                    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
                    "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
                ]
                for col in binary_cols:
                    if col in df_b.columns:
                        df_b[col] = df_b[col].map({"Yes": 1, "No": 0,
                                                    "No phone service": 0,
                                                    "No internet service": 0})

                cat_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]
                df_b = pd.get_dummies(df_b, columns=[c for c in cat_cols if c in df_b.columns],
                                      drop_first=False)
                bool_cols = df_b.select_dtypes(include="bool").columns
                df_b[bool_cols] = df_b[bool_cols].astype(int)

                # Engineered features — use the SAME thresholds saved during
                # training rather than a hardcoded/duplicated value.
                df_b["ChargesPerMonth"]  = df_b["TotalCharges"] / (df_b["tenure"] + 1)
                df_b["HighValue"]        = (df_b["MonthlyCharges"] > HIGH_VALUE_THRESHOLD).astype(int)
                df_b["LongTenure"]       = (df_b["tenure"] > LONG_TENURE_THRESHOLD).astype(int)
                svc_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                            "TechSupport", "StreamingTV", "StreamingMovies"]
                df_b["MultipleServices"] = df_b[[c for c in svc_cols if c in df_b.columns]].sum(axis=1)

                # Align columns
                for col in feature_columns:
                    if col not in df_b.columns:
                        df_b[col] = 0
                df_b = df_b[feature_columns]

                # Scale
                num_features = ["tenure", "MonthlyCharges", "TotalCharges",
                                "ChargesPerMonth", "MultipleServices"]
                df_b[num_features] = scaler.transform(df_b[num_features])

                probs   = model.predict_proba(df_b)[:, 1]
                preds   = (probs >= 0.5).astype(int)
                labels  = ["High" if p >= 0.70 else "Medium" if p >= 0.40 else "Low"
                           for p in probs]

                results_df = pd.DataFrame({
                    "CustomerID":        cust_ids,
                    "ChurnProbability":  np.round(probs, 4),
                    "Prediction":        ["Churn" if p == 1 else "No Churn" for p in preds],
                    "RiskLevel":         labels,
                })

                st.success(f"✅ Predictions complete for {len(results_df)} customers.")
                st.dataframe(results_df, use_container_width=True)

                # Summary
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Customers",  len(results_df))
                c2.metric("Predicted Churners", int(preds.sum()))
                c3.metric("Avg Churn Prob",   f"{probs.mean()*100:.1f}%")

                # Download
                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    csv_out, "churn_predictions.csv", "text/csv"
                )

            except Exception as e:
                st.error(f"Error during batch prediction: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Model Info
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Info":
    st.subheader("📊 Model Information & Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏗️ Pipeline Overview")
        st.markdown(f"""
| Step | Detail |
|---|---|
| **Dataset** | IBM Telco Customer Churn |
| **Rows** | 7,043 |
| **Features** | 30+ (after engineering) |
| **Target** | Churn (binary) |
| **Class Imbalance** | ~26% churn → balanced with SMOTE |
| **Train / Test Split** | 80% / 20% |
| **Cross-Validation** | Stratified 5-Fold |
| **Best Model** | {best_model_name} |
| **Tuning** | GridSearchCV |
| **Explainability** | SHAP TreeExplainer |
        """)

    with col2:
        st.markdown("### 📈 Typical Performance Benchmarks")
        st.markdown("""
| Metric | Score |
|---|---|
| **ROC-AUC** | ~0.85 |
| **F1 Score** | ~0.62 |
| **Accuracy** | ~80% |
| **Precision** | ~0.66 |
| **Recall** | ~0.58 |
        """)
        st.info(
            "Actual scores depend on the best model selected during training. "
            "Run the notebook to see exact figures."
        )

    st.markdown("### 🔑 Key Churn Drivers (from SHAP analysis)")
    st.caption(
        "Illustrative values shown below. Run the notebook's SHAP cell on "
        "your own trained model to get exact, data-driven importances."
    )
    drivers = {
        "Contract Type (Month-to-month)": 0.92,
        "Tenure (short)":                 0.85,
        "InternetService (Fiber optic)":  0.78,
        "MonthlyCharges (high)":          0.72,
        "OnlineSecurity (No)":            0.65,
        "TechSupport (No)":               0.60,
        "PaperlessBilling (Yes)":         0.55,
        "PaymentMethod (e-check)":        0.50,
    }
    df_drivers = pd.DataFrame.from_dict(
        drivers, orient="index", columns=["Importance"]
    ).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(9, 4))
    df_drivers["Importance"].plot(kind="barh", ax=ax, color="#4f8ef7")
    ax.set_title("Top Churn Drivers (illustrative SHAP importance)", fontsize=13)
    ax.set_xlabel("Relative Importance")
    st.pyplot(fig)

    st.markdown("""
---
**Made with ❤️ using Streamlit · scikit-learn · XGBoost · SHAP**
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#888;font-size:0.85rem;'>"
    "Customer Churn Predictor &nbsp;|&nbsp; Built with Streamlit &nbsp;|&nbsp; "
    "Model: XGBoost / Random Forest + SHAP</center>",
    unsafe_allow_html=True,
)
