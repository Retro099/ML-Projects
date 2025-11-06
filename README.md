# Customer Churn Prediction

## Project Overview
This project predicts the likelihood of customer churn using historical data.
It helps businesses identify at-risk customers and plan retention measures.

## Objective
Build a machine learning model that predicts churn with useful accuracy and recall.  
Model: Logistic Regression  
Dataset: Bank Customer Data (sample anonymized)

## Methodology
1. Data Preprocessing: missing values, categorical encoding, feature scaling  
2. EDA: correlations, churn trends, demographic insights  
3. Modeling: compared several classifiers, finalized logistic regression  
4. Evaluation Metrics: Accuracy, Recall, Precision, F1-score  

## Results
| Metric | Value |
|--------|--------|
| Accuracy | 0.82 |
| Recall | 0.57 |
| Precision | 0.65 |
| F1 Score | 0.60 |

All metrics are stored in results/metrics/metrics.json.

## Visuals
![Confusion Matrix](results/assets/confusion_matrix.png)  
![Feature Importance](results/assets/feature_importance.png)

## Business Insights
- Most churned customers had lower balance and shorter tenure.  
- Focus retention campaigns on customers aged 30-45 with 1-3 products.  

## Project Structure
ML-Projects/
  └── Customer_Churn_Prediction/
      ├── notebooks/
      ├── src/
      ├── data_sample/
      ├── results/
      │   ├── metrics/
      │   └── assets/
      └── README.md

## Tech Stack
Python, Pandas, Scikit-learn, Matplotlib, SHAP

## How to Run
1. Clone repo  
2. Open notebook in Colab or local Jupyter  
3. Run all cells end-to-end  

## Outputs
- Metrics JSON  
- Visual charts and SHAP plots  
- Trained model (optional)  

## Version
v1.0 - Stage 10 (Final Documentation and Polish)
