# Japanese Sentiment Analysis (NLP)

**Status:** Day 2 complete – Baseline training + metrics + Streamlit test script  
**Duration so far:** 2/9–10 days  
**Model:** cl-tohoku/bert-base-japanese-v2 (3-class)  
**Dataset:** WRIME ver2 (10k slice)  
**Weighted F1 Score:** **0.7507**

## 日本語要約
日本語のレビューやテキストに対する感情分析モデルを構築中です。Hugging Face Transformers + PyTorchを使用してBERTをファインチューニングし、Positive / Negative / Neutralの3クラス分類を実現。eコマース顧客満足度分析や自動レビュー分類に活用可能です。将来的にStreamlit Cloudでデプロイ予定です。

## Day 2 Achievements
- Dataset preprocessing + 3-class mapping (`avg_readers`)
- Tokenizer + fine-tuning with Trainer API (2 epochs)
- Confusion matrix + Weighted F1 = **0.7507**
- Streamlit `app.py` created (ready for deployment)

## Next Steps
Day 3: Model improvement, Gradio demo, full Streamlit Cloud deployment

**Assets:**  
- `assets/day2_confusion_matrix.png`  
- `app.py` (test script)

**Requirements:** See `requirements_nlp.txt`