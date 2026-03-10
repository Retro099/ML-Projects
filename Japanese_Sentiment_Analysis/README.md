# Japanese Sentiment Analysis (NLP)

**Status:** Day 3 complete – Gradio Live Demo + Production-ready Streamlit app  
**Duration:** 3/9–10 days  
**Model:** cl-tohoku/bert-base-japanese-v2 (fine-tuned)  
**Dataset:** Custom Japanese reviews (10k slice)  
**Weighted F1 Score (Day 2):** 0.7507  

## 日本語要約
日本語のレビューやテキストに対する感情分析モデルを構築しました。Hugging Face Transformers + PyTorchを使用してBERTをファインチューニングし、Positive / Negative / Neutralの3クラス分類を実現。eコマース顧客満足度分析や自動レビュー分類に活用可能です。現在GradioデモとStreamlit Cloudデプロイに対応しています。

## 🎯 Key Achievements
- **Day 1:** Baseline inference with Japanese BERT
- **Day 2:** Full fine-tuning + metrics (F1 0.7507) + confusion matrix
- **Day 3:** Gradio live demo + production `app.py` for Streamlit Cloud

## 🚀 Live Demos
- **Gradio Live Demo:** [https://f50c787d7b105f7bf9.gradio.live/](https://f50c787d7b105f7bf9.gradio.live/)
- **Streamlit Cloud:** Coming in Day 4 (app.py already ready)

## 📁 Project Structure
Japanese_Sentiment_Analysis/
├── notebooks/
├── assets/                  ← screenshots & plots
├── app.py                   ← Streamlit deployment file
├── requirements_nlp.txt
└── README.md