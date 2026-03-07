# Japanese Sentiment Analysis (NLP)

**Status:** Stage 0 – Skeleton ready  
**Duration:** 9–10 days  
**Goal:** Production-ready 3-class Japanese sentiment classifier (Positive / Negative / Neutral) for e-commerce reviews.

## 日本語要約
日本語のレビューやテキストに対する感情分析モデルを構築します。Hugging Face Transformers + PyTorchでBERTをファインチューニングし、Positive/Negative/Neutralの3クラス分類を実現。eコマース顧客満足度分析や自動レビュー分類に活用可能です。将来的にStreamlit Cloudでデプロイ予定。

## Deployment Plan
- Streamlit Cloud (app.py included from Day 3)
- Model saved with HF save_pretrained (zero version issues)
- Pinned requirements.txt

**Next:** Day 1 Colab notebook after this push.