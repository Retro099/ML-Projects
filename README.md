# 🔥 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Ready-success)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Retro099/ML-Projects/blob/main/Customer_Churn_Prediction/notebooks/Customer_Churn_Real.ipynb)

## 🚀 Live Demo
**[Try the Churn Predictor here](https://ml-projects-njqzlxkffdz9kzztmaszak.streamlit.app/)**

---

## 🧠 Project Overview
End-to-end churn prediction model using Telco customer data.  
Helps businesses target retention efforts on high-risk customers.

**Final Model**: Logistic Regression Pipeline  
**Dataset**: ~7,000 Telco records  
**Key Metrics**: Accuracy ~82%, Recall ~57%

## 📊 Results Summary
| Metric    | Value | Notes                          |
|-----------|-------|--------------------------------|
| Accuracy  | 0.82  | Overall correctness            |
| Recall    | 0.57  | Churn detection rate (priority)|
| Precision | 0.65  | Trust in positive predictions  |
| F1 Score  | 0.60  | Balance precision & recall     |

## 🖼️ Visuals
![Final Confusion Matrix](Customer_Churn_Prediction/assets/FINAL_CONFUSION_MATRIX.png)

## 🧮 Business Insights
- Strongest churn drivers: high monthly charges, short tenure, low balance  
- Target: 30–45 age group with 1–3 products

## 日本語概要 (Japanese Summary)
テレコ顧客データを用いた離職予測モデルを構築しました。  
精度82%、再現率57%を達成。主な離職要因は高額料金・短期間契約です。  
データ前処理からモデル評価、ビジネスインサイトまでエンドツーエンド実装済み。

## ▶️ How to Run
**Colab**: Click badge above  
**Local**: `streamlit run app.py`

## 🔄 Next Steps
- Add SHAP explanations  
- Periodic retraining

All artifacts in `artifacts/`. All visuals in `assets/`.
