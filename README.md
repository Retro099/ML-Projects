# AI/ML Engineer Portfolio
Python | Scikit-learn | PyTorch | Hugging Face | RAG | Docker  

Open to mid-level AI/ML roles  
Each project has its own pinned `requirements.txt` — see individual folders for exact dependencies and how to run.

---

### ✅ Completed Projects

#### 1. Japanese RAG Production System
**Status:** Live demo — overlap-dense retrieval + Groq Qwen  

**Key Achievements**  
- Modular RAG for Japanese PDFs: PyMuPDF → JP chunking (overlap 50) → bge-m3 → Chroma → FastAPI + Streamlit + Docker  
- Live generator: Groq `qwen/qwen3.6-27b` (`reasoning_effort=none`). Old Groq 70B is retired  
- Overlap rebuild put 楽天 Non-GAAP **1,063億円** and **トリプル20** at dense rank 1  
- Hybrid + 3 JP rerankers tried and not shipped (they dropped the 1,063 factoid)  
- Two evals: v1 notebook 0.96/0.76 (70B, curated contexts) vs 6 Sep live `/ask` dump (F=1.0 on 4 finished rows, AR ≈ 0.62; 2 long rows judge-truncated)  

**Folder:** [Japanese_RAG_Production](./Japanese_RAG_Production)

---

#### 2. Credit Card Fraud Detection  
**Status:** ✅ COMPLETED & PRODUCTION-READY  

**Key Achievements**  
- High-recall XGBoost (recall 0.92, PR-AUC 0.85)  
- SHAP explainability (V14/V17 main drivers)  
- Docker container + Streamlit live demo  
- Unit tests + pinned dependencies  
- Business insights included  

**Live Demo (Streamlit Cloud):** [https://ml-projects-credit-card-fraud-detection.streamlit.app/]  

---

#### 3. Japanese Sentiment Analysis (NLP)  
**Status:** ✅ COMPLETED & PORTFOLIO-READY  

**Key Achievements**  
- Fine-tuned Japanese BERT (cl-tohoku/bert-base-japanese-v2) with 3-class sentiment  
- Production deployment on Gradio + Streamlit Cloud (CPU-optimized)  
- Model pushed to Hugging Face Hub (Retro099/japanese-sentiment-analysis-v1)  
- Professional assets: confusion matrix + documentation  

**Live Demo:** Gradio → https://f50c787d7b105f7bf9.gradio.live/  
**Streamlit Cloud:** [https://cx7v54eehcppwnarlaplxt.streamlit.app/]  
**Model on HF Hub:** https://huggingface.co/Retro099/japanese-sentiment-analysis-v1

---

#### 4. Customer Churn Prediction
**Status:** ✅ COMPLETED & LIVE  

**Key Achievements**  
- End-to-end ML pipeline with production-ready artifact  
- Interactive Streamlit web application  
- Strong business insights and documentation  
- Accuracy 0.82 | Recall 0.57 (priority metric)

**Live Demo:** [Streamlit App](https://ml-projects-njqzlxkffdz9kzztmaszak.streamlit.app/)

---

**All projects follow PEP8 standards, modular structure, and pinned dependencies.**  
Every project includes clear documentation and business impact section.
