import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Fraud Detector", page_icon="🚨", layout="centered")

st.title("🚨 Credit Card Fraud Detection")
st.markdown("**High-recall XGBoost model** – V14 & V17 are the strongest drivers (SHAP)")

COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

@st.cache_resource
def load_model():
    MODEL_PATH = Path(__file__).parent / "artifacts" / "fraud_model_v1.joblib"
    return joblib.load(MODEL_PATH)

model = load_model()

col1, col2 = st.columns(2)
with col1:
    time = st.number_input("Time (seconds)", value=0.0, step=1.0)
with col2:
    amount = st.number_input("Amount (USD)", value=100.0, step=1.0)

with st.expander("V1–V28 Features", expanded=False):
    v_values = [st.number_input(f"V{i}", value=0.0, step=0.01, key=f"v{i}") for i in range(1, 29)]

if st.button("🔍 Predict Fraud", type="primary"):
    input_data = [time, amount] + v_values
    df_input = pd.DataFrame([input_data], columns=COLUMNS)
    prob = model.predict_proba(df_input)[0][1]
    
    st.metric("Fraud Probability", f"{prob:.4f}")
    
    if prob > 0.5:
        st.error("🚨 HIGH RISK – BLOCK TRANSACTION")
    else:
        st.success("✅ Normal transaction – Proceed safely")