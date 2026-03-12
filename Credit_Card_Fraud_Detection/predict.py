import joblib
from pathlib import Path
import numpy as np

MODEL_PATH = Path("artifacts/fraud_model_v1.joblib")
model = joblib.load(MODEL_PATH)

def predict_fraud(features: list) -> dict:
    input_array = np.array(features).reshape(1, -1)
    prob = model.predict_proba(input_array)[0][1]
    return {
        "fraud_probability": float(prob),
        "is_fraud": bool(prob > 0.5),
        "recommendation": "🚨 BLOCK TRANSACTION" if prob > 0.5 else "✅ Normal"
    }

# Test
if __name__ == "__main__":
    sample = [0, 100, *np.random.randn(28)]  # dummy input
    print(predict_fraud(sample))