# Credit Card Fraud Detection

High-recall fraud detection model on imbalanced credit card transactions (0.172% fraud rate).  
Focus: extreme class imbalance handling, SHAP explainability, business recommendations for Japanese fintech.

## Key Results
- Best model: XGBoost with scale_pos_weight  
- Recall (fraud): ~0.92 (very high – minimizes financial loss)  
- PR-AUC: ~0.85  
- Top drivers (SHAP): V14, V17 (negative values strongly indicate fraud)

## Model Comparison
| Model          | Recall (Fraud) | PR-AUC   | Notes                          |
|----------------|----------------|----------|--------------------------------|
| Logistic + weight | 0.78        | 0.72     | Day 3 baseline                 |
| RandomForest   | 0.88           | 0.80     | balanced_subsample             |
| XGBoost        | **0.92**       | **0.85** | Best – scale_pos_weight        |
| LightGBM       | 0.90           | 0.83     | Fast & competitive             |

## SHAP Explainability
V14とV17が不正取引の主要ドライバーであることがSHAP解析で明らかになりました。  
低いV14値は不正確率を強く押し上げ、リアルタイム監視ルールに活用可能です。

## Business Recommendations
- リアルタイム監視システムにV14/V17のSHAP値を組み込み、閾値超過で即時アラート発行
- 偽陰性コストが高いため、再現率0.90以上を優先運用
- モデルドリフト検知時に自動再学習トリガー
- 顧客説明用に「異常検知根拠」を簡易表示

## プロジェクト概要（日本語サマリー）
深刻なクラス不均衡のクレジットカード取引データに対し、高再現率の不正検知モデルを構築しました。  
XGBoost + SHAPにより、再現率0.92を達成し、V14/V17が不正の主要指標であることを解明。  
偽陰性による損失を大幅に低減し、金融機関の信頼性向上に貢献します。

## 🚀 Production Deployment (Docker)

### How to run locally
```bash
docker compose up --build
```

### How to run Locally
```bash
python -m pytest tests/test_model.py -v
```

### Production-ready features
Pinned dependencies (scikit-learn==1.6.1)
Unit tests included
Single predict.py endpoint ready for API
Model loaded from artifacts/fraud_model_v1.joblib
High-recall XGBoost + SHAP explainability

## 日本語概要 (Production版)
不均衡なクレジットカード不正検知データに対し、高再現率のXGBoostモデルを構築（再現率0.92、PR-AUC 0.85）。SHAP解析でV14/V17が主要ドライバーと判明。Dockerコンテナ + ユニットテスト完備で本番即運用可能。銀行実務レベルのケーススタディです。