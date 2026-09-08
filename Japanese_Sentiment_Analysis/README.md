# Japanese Sentiment Analysis

Fine-tuned Japanese BERT for 3-class sentiment (positive / neutral / negative).

**Model:** [Retro099/japanese-sentiment-analysis-v1](https://huggingface.co/Retro099/japanese-sentiment-analysis-v1)

**Live demo:** [Streamlit Cloud](https://cx7v54eehcppwnarlaplxt.streamlit.app/)

## What it does
- Base model: `cl-tohoku/bert-base-japanese-v2`
- End-to-end fine-tune + CPU inference (no GPU required to run the demo)
- Model card on Hugging Face Hub

Useful for routing Japanese reviews or support text without a human reading every line.

## Stack
PyTorch · Hugging Face Transformers · Streamlit · `fugashi` / `unidic-lite`