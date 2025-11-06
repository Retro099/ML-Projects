import json, joblib, numpy as np, pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DATA_PATH     = Path(__file__).resolve().parent / "data_sample" / "Customer_Churn.csv"
METRICS_JSON  = Path(__file__).resolve().parent / "results" / "metrics" / "metrics_deploy.json"

@st.cache_resource
def load_pipeline():
    pipe = joblib.load(ARTIFACTS_DIR / "churn_pipeline.joblib")
    return pipe

@st.cache_data
def load_manifest():
    with open(ARTIFACTS_DIR / "manifest.json") as f:
        return json.load(f)

@st.cache_data
def load_metrics():
    try:
        with open(METRICS_JSON) as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_data_preview():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception:
        return None

pipe = load_pipeline()
manifest = load_manifest()
metrics = load_metrics()
df_preview = load_data_preview()

num_cols = manifest.get("columns", {}).get("numeric", [])
cat_cols = manifest.get("columns", {}).get("categorical", [])

st.title("Customer Churn Prediction")
st.caption("Logistic Regression pipeline • unified preprocessing + model")

# Sidebar: deployment metrics
with st.sidebar:
    st.header("Model Snapshot")
    st.write(f"Artifact: `{manifest.get('artifact', 'N/A')}`")
    st.write(f"scikit-learn: `{manifest.get('sklearn_version', 'N/A')}`")
    if metrics:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}")
        st.metric("Recall (Yes)", f"{metrics.get('recall', 0):.2f}")
        st.metric("Precision (Yes)", f"{metrics.get('precision', 0):.2f}")
        st.metric("F1 (Yes)", f"{metrics.get('f1', 0):.2f}")
    st.markdown("---")
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.01)

# Helper: infer categorical choices
def infer_cat_choices(col):
    if df_preview is not None and col in df_preview.columns:
        vals = df_preview[col].dropna().unique().tolist()
        vals = [v for v in vals if v != ""]
        if len(vals) > 0:
            return sorted(map(str, vals))
    # fallback: try encoder categories
    try:
        pre = pipe.named_steps.get("preprocessor", None)
        if pre and hasattr(pre, "transformers_"):
            for name, trans, cols in pre.transformers_:
                if hasattr(trans, "categories_") and col in cols:
                    idx = cols.index(col)
                    return list(map(str, trans.categories_[idx].tolist()))
    except Exception:
        pass
    return ["Unknown"]

# Helper: infer numeric min/max
def infer_num_range(col):
    if df_preview is not None and col in df_preview.columns:
        s = pd.to_numeric(df_preview[col], errors="coerce").dropna()
        if s.size:
            lo, hi = float(np.nanpercentile(s, 1)), float(np.nanpercentile(s, 99))
            if lo == hi:
                hi = lo + 1.0
            return lo, hi
    return 0.0, 100000.0

# Build input UI
st.subheader("Inputs")
cols_layout = st.columns(2)

inputs = {}

# numeric
for i, col in enumerate(num_cols):
    lo, hi = infer_num_range(col)
    step = max((hi - lo) / 100.0, 0.01)
    with cols_layout[i % 2]:
        val = st.number_input(col, value=float((lo + hi) / 2.0), min_value=float(lo), max_value=float(hi), step=float(step))
    inputs[col] = val

# categorical
for i, col in enumerate(cat_cols):
    choices = infer_cat_choices(col)
    with cols_layout[i % 2]:
        val = st.selectbox(col, options=choices, index=0)
    inputs[col] = val

# Assemble single-row DataFrame as raw features
X_row = pd.DataFrame([inputs])

# Detect positive class index robustly
def positive_index(model):
    classes = getattr(model, "classes_", None)
    if classes is None:
        return 1
    # prefer 'Yes' if present, else max class
    if "Yes" in classes: 
        return int(np.where(classes == "Yes")[0][0])
    try:
        return int(np.argmax(classes))
    except Exception:
        return 1

if st.button("Predict"):
    try:
        probas = pipe.predict_proba(X_row)[0]
        pos_idx = positive_index(pipe.named_steps["model"])
        proba = float(probas[pos_idx])
        label = "Likely to Churn" if proba >= threshold else "Not Likely to Churn"
        st.metric("Churn Probability", f"{proba:.2%}")
        st.write(label)
        st.progress(min(max(proba, 0.0), 1.0))
        st.caption("Decision uses the threshold in the sidebar.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Tip: adjust the decision threshold based on business costs of false positives vs false negatives.")
