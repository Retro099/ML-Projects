# Customer Churn Prediction — One-Page Summary

## Problem
Identify customers likely to churn so the business can act before revenue loss.

## Data
Bank customer dataset (sample). Cleaned, encoded, and scaled. Balanced checks performed.

## Approach
- Models tested: Logistic Regression vs tree-based baselines
- Final: Logistic Regression for stable recall and interpretability
- End-to-end notebook with reproducible steps

## Key Metrics
- Accuracy: 0.82
- Recall: 0.57
- Precision: 0.65
- F1-score: 0.60
(also stored in results/metrics/metrics.json)

## Insights (Actionable)
- Higher churn risk for short-tenure, low-balance customers
- Target 30–45 age band with 1–3 products for retention upsell
- Focus on proactive outreach and fee relief tests

## What a client gets
- Clean notebook and results
- Metrics JSON for tracking
- Ready visuals for reports
- Clear README and version tag (v1.0)

## How to Run
1) Open Customer_Churn_Real.ipynb in Colab
2) Run all cells
3) See results in results/metrics and results/assets

## Next Steps (Deployment)
- Streamlit mini-app to score new customers
- Optional SHAP plots page for explainability
