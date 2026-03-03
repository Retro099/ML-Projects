# app.py – Simplified & Reliable Version (pandas.get_dummies)
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("🔥 Customer Churn Predictor")
st.markdown("Telco Customer Churn Demo – Japan 2026 Portfolio")

# Train model on startup (cached, fast)
@st.cache_resource
def train_model():
    # Robust path that works on Streamlit Cloud
    csv_path = Path(__file__).parent / "data_sample" / "Customer_Churn.csv"
    df = pd.read_csv(csv_path)
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop(columns=["Churn_Yes"])
    y = df["Churn_Yes"]
    
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)  # full data for demo
    
    return model, X.columns.tolist()

model, feature_cols = train_model()
st.success("✅ Model ready!")

# Simple form
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", 0, 72, 24)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

if st.button("Predict Churn", type="primary"):
    input_row = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner_Yes": 1 if partner == "Yes" else 0,
        "Dependents_Yes": 1 if dependents == "Yes" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,
        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No": 1 if internet == "No" else 0,
        "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
        "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
    }
    
    df_input = pd.DataFrame([input_row])
    
    # Add missing columns with 0
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    
    df_input = df_input[feature_cols]
    
    prob = model.predict_proba(df_input)[0][1]
    
    st.subheader("Result")
    st.metric("Churn Probability", f"{prob:.1%}")
    
    if prob >= 0.5:
        st.error("**High Risk** → Recommend immediate retention action")
    else:
        st.success("**Low Risk** → Customer likely to stay")

st.caption("Simplified version • Ready for Streamlit Cloud • Japan 2026 Portfolio")