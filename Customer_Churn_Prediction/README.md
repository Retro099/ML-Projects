# Customer Churn Prediction

Telco churn baseline. Logistic regression pipeline on ~10k anonymized customer rows.

**Metrics:** Accuracy 0.82 · Recall 0.57 · Precision 0.65 · F1 0.60  
Recall is the priority (catch actual churners). It is not a high-recall model.

**Live demo:** [Streamlit](https://ml-projects-njqzlxkffdz9kzztmaszak.streamlit.app/)

![Confusion matrix](./assets/FINAL_CONFUSION_MATRIX.png)

Details: [`artifacts/manifest.json`](./artifacts/manifest.json)

## Notes
- Stronger churn signals in this set: higher monthly charges, short tenure
- This is a baseline demo, not a production scoring service

## Run locally
```bash
git clone https://github.com/Retro099/ML-Projects.git
cd ML-Projects/Customer_Churn_Prediction
pip install -r requirements.txt
streamlit run app.py
```

Notebook (training trail): `notebooks/Customer_Churn_Real.ipynb`