FROM python:3.10-slim

WORKDIR /app

COPY requirements_fraud.txt .
RUN pip install --no-cache-dir -r requirements_fraud.txt

# Copy BOTH files so both services work
COPY app.py .
COPY predict.py .
COPY artifacts/fraud_model_v1.joblib ./artifacts/

EXPOSE 8501

# Default is Streamlit (demo), but we can run CLI too
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]