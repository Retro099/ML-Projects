# AI/ML Engineer Portfolio

Python · PyTorch · Hugging Face · RAG · Docker · FastAPI

Open to mid-level AI/ML roles. Each folder has its own `requirements.txt`.

### Japanese document Q&A (RAG)

Ask questions against Japanese company PDFs. The system has to retrieve the right passage and not invent figures.

- Japanese-aware chunking with overlap → bge-m3 → Chroma → FastAPI + Streamlit + Docker
- After the overlap rebuild, Rakuten’s Non-GAAP operating profit (1,063億円) and the “トリプル20” AI line rank first
- Extra rerankers were tried and left out — they pushed that profit figure down
- Folder: [Japanese_RAG_Production](./Japanese_RAG_Production)

### Credit card fraud

XGBoost on a heavily imbalanced set. Fraud recall **0.92**, PR-AUC **0.85**. SHAP: V14 / V17. Docker + tests.

[Live demo](https://ml-projects-credit-card-fraud-detection.streamlit.app/) · [Folder](./Credit_Card_Fraud_Detection)

### Japanese sentiment

Fine-tuned `cl-tohoku/bert-base-japanese-v2` (positive / neutral / negative). CPU demo + model on Hugging Face.

[Streamlit](https://cx7v54eehcppwnarlaplxt.streamlit.app/) · [HF model](https://huggingface.co/Retro099/japanese-sentiment-analysis-v1) · [Folder](./Japanese_Sentiment_Analysis)

### Customer churn

Telco churn baseline. Accuracy 0.82, recall 0.57. Live form only.

[Live demo](https://ml-projects-njqzlxkffdz9kzztmaszak.streamlit.app/) · [Folder](./Customer_Churn_Prediction)