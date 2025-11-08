# Customer Churn Prediction
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Retro099/ML-Projects/blob/main/Customer_Churn_Prediction/notebooks/Customer_Churn_Real.ipynb)

---

## 🧩 Overview
Predicts which telecom customers are likely to churn so teams can act before they leave.

---

## 📊 Dataset
- **Source:** Telco Customer Churn dataset (Kaggle or custom)
- **Target:** `Churn` (0 = stay, 1 = leave)
- **Processing:** Encoded categoricals, handled nulls, normalized numericals.

---

## 🤖 Models & Metrics
| Model | Accuracy | Precision | Recall | F1 |
|--------|-----------|------------|---------|----|
| Logistic Regression | 0.79 | 0.63 | 0.52 | 0.57 |
| Random Forest (final) | **0.82** | **0.68** | **0.57** | **0.59** |

---

## 🧠 Tech Stack
Python • pandas • scikit-learn • matplotlib • seaborn • Colab ↔ GitHub

---

## ▶️ Run the Project
**In Colab:**
Click the badge above.

**Locally:**
```bash
git clone https://github.com/Retro099/ML-Projects.git
cd ML-Projects/Customer_Churn_Prediction
pip install -r requirements.txt
jupyter notebook notebooks/Customer_Churn_Real.ipynb


## Results
Add visuals to `assets/` and reference them here:
![Confusion Matrix](Customer_Churn_Prediction/assets/FINAL_CONFUSION_MATRIX.png)
![Random Forest](Customer_Churn_Prediction/assets/RANDOM_FOREST_CONFUSION_MATRIX.png)

The model helps target high-risk customers for proactive retention, saving marketing resources and improving customer lifetime value.
