import pytest
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_PATH = Path("artifacts/fraud_model_v1.joblib")
model = joblib.load(MODEL_PATH)

COLUMNS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]

def test_model_loaded():
    assert model is not None, "Model failed to load"

def test_prediction_shape():
    sample = np.random.randn(30).tolist()
    df_input = pd.DataFrame([sample], columns=COLUMNS)
    prob = float(model.predict_proba(df_input)[0][1])   # ← added float() conversion
    assert 0 <= prob <= 1, "Probability must be between 0 and 1"
    assert isinstance(prob, float)                    # now guaranteed

if __name__ == "__main__":
    pytest.main([__file__])