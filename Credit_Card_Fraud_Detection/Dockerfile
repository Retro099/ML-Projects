FROM python:3.10-slim

WORKDIR /app

COPY requirements_fraud.txt .
RUN pip install --no-cache-dir -r requirements_fraud.txt

COPY predict.py .
COPY artifacts/fraud_model_v1.joblib ./artifacts/

EXPOSE 8080

CMD ["python", "predict.py"]