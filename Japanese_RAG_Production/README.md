# Japanese RAG Production System
**Status:** Live demo — overlap-dense retrieval + Groq Qwen generator

**Production-Ready Retrieval-Augmented Generation Pipeline for Japanese Documents**

A modular, evaluation-aware RAG system focused on Japanese text, featuring clean architecture, real RAGAS evaluation, FastAPI backend, and a professional Streamlit interface.

## 1. Project Overview

This project is a mid-level production-oriented RAG system designed for Japanese documents. It was built to demonstrate practical engineering skills relevant to the 2026 job market in India.

**Goal:**  
Build a clean, modular, and measurable RAG pipeline that properly handles Japanese text characteristics (no spaces, long sentences, compound words) while following production engineering practices.

**Why this project matters:**

- Japanese document handling is still a real differentiator for Japan-targeted roles.
- Many portfolios stop at basic RAG tutorials. This project goes further with evaluation, API design, and production thinking.

## 2. Key Features

- **Japanese-aware chunking** with configurable size and overlap
- **bge-m3 embeddings** optimized for multilingual (including Japanese) retrieval
- **Two evals:** v1 notebook (70B judge, curated contexts) and 6 Sep live /askdump (seeevaluation/)
- **API-first architecture** (FastAPI backend + Streamlit frontend)
- **Professional Streamlit UI** with latency metrics, grounding status, and source citations
- **Docker-ready** with `Dockerfile` and `docker-compose.yml`
- Modular codebase following clean separation of concerns

---

## 3. Tech Stack

| Component        | Technology                     | Reason                                             |
| ---------------- | ------------------------------ | -------------------------------------------------- |
| Embeddings       | BAAI/bge-m3                    | Strong multilingual performance, good for Japanese |
| Vector Database  | ChromaDB                       | Lightweight and easy to use for local development  |
| LLM              | Groq qwen/qwen3.6-27b (reasoning_effort=none) | Live generator. llama-3.3-70b-versatile retired on Groq 16 Aug 2026        |
| Backend          | FastAPI + Uvicorn              | Clean API layer and production readiness           |
| Frontend         | Streamlit                      | Fast interactive demo                              |
| Evaluation       | RAGAS                          | Industry-standard RAG metrics                      |
| Containerization | Docker + docker-compose        | Production deployment support                      |

---

## 4. Architecture & Design Decisions
```

PDF Documents
↓
Document Ingestion (PyMuPDF)
↓
Japanese-aware Chunking
↓
bge-m3 Embeddings → ChromaDB
↓
Retriever (Top-K)
↓
Generator (Groq + Improved Prompt)
↓
FastAPI Backend ←→ Streamlit Frontend
```

**Business / Production notes:**
- Real RAGAS evaluation was run on actual system outputs, not synthetic data.
- Modular design (ingestion → Japanese-aware chunking → retrieval → generation) makes the pipeline easy to adapt to other languages or domain documents.

**Key Design Decisions:**

- **Modular structure** (`src/` folder) so each component can be tested and improved independently.
- **API-first approach**: Streamlit does not contain business logic. It only calls the FastAPI backend.
- **Real evaluation**: Used actual outputs from the system for RAGAS instead of synthetic data.
- **Prompt engineering**: Multiple iterations were done to improve answer relevancy while keeping faithfulness high.
- **Docker support**: Prepared for production environments even though local testing was limited by hardware constraints.

---

## 5. Screenshots & Demo

> **Live Demo:** Run locally using the instructions below
> (Streamlit UI + FastAPI backend)
![Streamlit Demo](assets/RAG1.png)

**Main Interface:**
- Clean dark theme
- Latency metrics (Retrieval Time + Generation Time)
- Grounding status indicator
- Expandable source citations

---

## 6. Setup & Installation

### Prerequisites
- Python 3.10+
- Groq API Key

### Local Development (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Japanese_RAG_Production.git
cd Japanese_RAG_Production
```

```

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env
# Add your GROQ_API_KEY inside .env

# 5. Start FastAPI backend
python api.py

# 6. In another terminal, start Streamlit
streamlit run app/streamlit_app.py

### Production Deployment (Docker)

```bash
# 1. Create .env file
cp .env.example .env
# Add your GROQ_API_KEY

# 2. Build and run
docker-compose up --build
```

This will start:

- FastAPI backend → `http://localhost:8000`
- Streamlit frontend → `http://localhost:8501`

---

## 7. Evaluation Results

**v1 notebook** (`evaluation/ragas_evaluation.ipynb`): curated contexts, judge `llama-3.3-70b-versatile`.
Faithfulness 0.9643 / Answer Relevancy 0.7633. Not the live stack.

**Live 6 Sep** (`evaluation/RESULTS_2026-09-06.md`, `ragas_live_qwen.json`):
overlap-dense top-5 + Qwen answers. Judge `openai/gpt-oss-120b` (Qwen cannot host RAGAS on Groq free OTPM).

| # | Faithfulness | Answer Relevancy |
|---|---|---|
| 4 finished factoid rows | 1.00 | — |
| 2 long-answer rows | judge truncated | — |
| Mean AR (6 rows) | — | ~0.62 |

Retrieval check: after overlap rebuild, 楽天 Non-GAAP 1,063億円 and トリプル20 are dense rank 1.
Hybrid BM25 and three JP rerankers (xsmall-v2, small-v2, ruri-v3-310m) were tried and **not shipped** — they saturated and dropped 1,063 from rank 1.

---

## 8. Limitations & Future Improvements

### Current Limitations

- Answer relevancy can still be improved for broader questions
- Docker setup was prepared for production but not fully tested in a high-resource environment due to local hardware constraints
- Limited number of documents in the current demo

### Planned Improvements

- Re-ranking was tried (3 models) and rejected on the 1,063 factoid. Live retriever stays dense-only.
- Expansion of RAGAS evaluation with more diverse test cases
- Full Docker deployment testing and CI/CD integration
- Further prompt and chunk quality improvements

---

## 9. Project Structure

```
Japanese_RAG_Production/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embedding_store.py
│   ├── retrieval.py
│   ├── generation.py
│   
├── data/
│   └── sample/
├── evaluation/
│   ├── ragas_evaluation.ipynb
│   ├── ragas_live_2026-09-06.ipynb
│   ├── ragas_live_qwen.json
│   └── RESULTS_2026-09-06.md
├── api.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## 10. Japanese Summary (日本就業に向けた技術的サマリー)

本プロジェクトでは、日本語文書を対象とした実用的なRAGシステムを構築しました。

### 技術的アプローチ

- `bge-m3`埋め込みモデルと日本語を考慮したチャンキング戦略を採用
- モジュール設計により保守性と拡張性を重視
- 評価は2系統。旧ノートブック（70B、0.96/0.76）と、2026年9月6日のライブ出力（Faithfulness は完了4件が1.00、Answer Relevancy平均約0.62）。ライブ生成は Qwen。
- FastAPIによるバックエンドとStreamlitによるフロントエンドを分離したAPI-firstアーキテクチャ

### 現在の成果と課題

- 回答の根拠性（Faithfulness）は非常に高く、幻覚の抑制に成功
- 回答の関連性（Answer Relevancy）は改善の余地があり、プロンプト最適化が今後の課題
- ローカル環境の制約によりDockerでの完全な動作確認は限定的だったが、本番展開用の設定ファイルは整備済み

### 今後の展望

- Dockerを活用した本番環境への展開
- RAGAS評価の自動化
- リランカーは試したが 1,063億円の順位が落ちたため未搭載。検索は overlap-dense のまま。

本プロジェクトは、日本語文書を扱う実務的なRAGシステムの構築経験と、評価・API設計といった中級レベルのエンジニアリングスキルを証明するものです。

---

## Lessons Learned

- Real evaluation is significantly more valuable than dummy data.
- Prompt engineering has a large impact on Answer Relevancy.
- Separating frontend and backend (even in a small project) greatly improves code quality and interview discussion points.
- Being honest about limitations is more professional than overclaiming.

---

**Note:** This project was developed with limited local computing resources. The core pipeline, evaluation, and API layer were fully implemented and tested. Docker configuration is provided for production environments with sufficient resources.
