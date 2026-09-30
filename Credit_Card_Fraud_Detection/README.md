# Credit Card Fraud Detection

High-recall fraud detection on heavily imbalanced credit-card transactions (0.172% fraud).  
Focus: class imbalance, SHAP explainability, and a runnable Docker demo.

## Key Results
- Best model: XGBoost with `scale_pos_weight`
- Recall (fraud): **0.84** · Precision: **0.88** (threshold 0.5)
- PR-AUC: **0.88**
- Top SHAP drivers: V14, V17 (low values push the score toward fraud)

## Model Comparison
| Model               | Recall (Fraud) | Precision | PR-AUC   | Notes                           |
|---------------------|----------------|-----------|----------|---------------------------------|
| Logistic (balanced) | 0.92           | 0.06      | n/a      | High recall, unusable precision |
| RandomForest        | 0.81           | n/a       | 0.82     | balanced_subsample              |
| XGBoost (shipped)   | **0.84**       | **0.88**  | **0.88** | scale_pos_weight, threshold 0.5 |
| LightGBM            | 0.84           | n/a       | 0.87     | Close second                    |

## SHAP
V14 and V17 are the strongest fraud drivers. Strongly negative V14 in particular lifts fraud probability and can be used as a simple monitoring rule.

## Business notes
- Prefer high recall over raw accuracy; a missed fraud costs more than an extra review.
- Next step: tune the threshold on a validation split to push recall toward 0.90 at a stated precision.
- If performance drifts, retrain; the Docker image is built to swap the `joblib` artifact.

## Docker

```bash
docker compose up --build
```

```bash
python -m pytest tests/test_model.py -v
```

Pinned `scikit-learn==1.6.1`, unit tests, `predict.py` CLI, model at `artifacts/fraud_model_v1.joblib`.

## Status
- XGBoost recall 0.84 / precision 0.88 / PR-AUC 0.88
- SHAP (V14 / V17)
- Docker (CLI + Streamlit) + tests

**Local demo:** http://localhost:8501 (`docker compose up fraud-web`)

**Streamlit Cloud:** [https://ml-projects-credit-card-fraud-detection.streamlit.app/](https://ml-projects-credit-card-fraud-detection.streamlit.app/)