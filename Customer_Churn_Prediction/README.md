# 🔥 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Ready-success)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Retro099/ML-Projects/blob/main/Customer_Churn_Prediction/notebooks/Customer_Churn_Real.ipynb)

---

## 🧠 Project Overview
End-to-end churn prediction model using bank customer data.  
Identifies high-risk customers for targeted retention campaigns.

**Final Model**: Logistic Regression Pipeline (full preprocessing + classification)  
**Dataset**: ~10,000 anonymized records  
**Key Metrics**: Accuracy ~82%, Recall ~57% (prioritizes catching actual churners)

## 📊 Results Summary

| Metric    | Value | Notes                          |
|-----------|-------|--------------------------------|
| Accuracy  | 0.82  | Overall correctness            |
| Recall    | 0.57  | Churn detection rate (priority)|
| Precision | 0.65  | Trust in positive predictions  |
| F1 Score  | 0.60  | Balance precision & recall     |

Full details & column info: [`artifacts/manifest.json`](./artifacts/manifest.json)

## 🖼️ Visuals
![Final Confusion Matrix](./assets/FINAL_CONFUSION_MATRIX.png)

(All plots & SHAP visuals saved in `assets/` folder)

## 🧮 Business Insights
- Strongest churn drivers: **high monthly charges**, **short tenure**, **low balance**  
- Recommended focus: Customers **aged 30–45** with **1–3 products**  
- Action: Proactive offers on top ~20% predicted risk → reduces revenue loss efficiently

## 日本語概要 (Japanese Summary)
銀行顧客データを用いた離職予測プロジェクトです。  
ロジスティック回帰パイプラインを採用し、精度82%、再現率57%を達成。  
主な離職要因は高額料金・短期間契約・低残高。  
30-45歳、1-3商品保有顧客への保持施策を提案。  
データ前処理からモデル評価、ビジネスインサイトまでエンドツーエンド実装。

## ▶️ How to Run
**Colab (recommended)**: Click badge above

**Local**:
```bash
git clone https://github.com/Retro099/ML-Projects.git
cd ML-Projects/Customer_Churn_Prediction
pip install -r requirements.txt
jupyter notebook notebooks/Customer_Churn_Real.ipynb