# Improving Domain-Specific Question Answering using Retrieval-Augmented Generation

> **Course:** DS8008 — Natural Language Processing | Toronto Metropolitan University  
> **Group 6:** Nishi Patel (501356244) · Avikumar Patel (501376903)  
> **Paper:** RetrievalQA: Assessing Adaptive RAG for Short-form Open-Domain QA (ACL Findings 2024)  
> **Paper Link:** https://arxiv.org/abs/2402.16457  
> **Dataset:** [hyintell/RetrievalQA](https://huggingface.co/datasets/hyintell/RetrievalQA) on HuggingFace

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Paper Summary](#paper-summary)
3. [What We Implemented](#what-we-implemented)
4. [Results](#results)
5. [Project Structure](#project-structure)
6. [How to Run](#how-to-run)
7. [Dependencies](#dependencies)
8. [Differences from the Original Paper](#differences-from-the-original-paper)
9. [References](#references)

---

## Project Overview

Large Language Models (LLMs) like GPT-3.5 often generate confident but factually incorrect answers for questions about recent or long-tail knowledge — a problem known as **hallucination**. This happens because LLMs rely solely on knowledge memorised during training, which has a fixed cutoff date.

This project implements and evaluates **Retrieval-Augmented Generation (RAG)** as a solution. RAG improves LLM accuracy by retrieving relevant documents from an external database and conditioning the model's answer on that evidence — similar to giving a student a reference book before an exam.

We evaluate **four retrieval strategies** on the RetrievalQA benchmark using **GPT-3.5-turbo**:

| Strategy | Description |
|----------|-------------|
| **No Retrieval** | GPT-3.5 answers from memory only |
| **Always Retrieval** | GPT-3.5 always receives retrieved context |
| **Adaptive (Oracle)** | Uses gold label to decide — upper-bound baseline |
| **TA-ARE** | GPT-3.5 decides itself using date + in-context examples |

---

## Paper Summary

**RetrievalQA: Assessing Adaptive Retrieval-Augmented Generation**  
Zhang, Fang & Chen — ACL Findings 2024

The paper asks: *Does an LLM even know when it needs to retrieve?*

The authors built a benchmark of **2,785 short-form QA questions** from five sources and introduced **TA-ARE (Time-Aware Adaptive REtrieval)** — a method that helps LLMs decide when to retrieve using:
- Today's date (for temporal reasoning)
- 2 YES + 2 NO in-context examples (to calibrate the decision)

**Key findings:**
- GPT-3.5 with vanilla prompting only made correct retrieval decisions **49.3%** of the time
- TA-ARE improved this to **86.3%** retrieval accuracy (+14.9% average across all LLMs)
- Always Retrieval consistently outperforms No Retrieval on knowledge-intensive questions

---

## What We Implemented

### Four Retrieval Strategies

**Strategy 1 — No Retrieval**
```
Answer = GPT-3.5(question)
```

**Strategy 2 — Always Retrieval**
```
Answer = GPT-3.5(question + top-3 retrieved passages)
```

**Strategy 3 — Adaptive Oracle**
```
if gold_label == "needs retrieval":
    Answer = GPT-3.5(question + context)
else:
    Answer = GPT-3.5(question)
```

**Strategy 4 — TA-ARE (Paper's method)**
```
decision = GPT-3.5("Today is Jan 2024. [2 yes / 2 no examples]. Do you need to retrieve?")
if decision == "yes":
    Answer = GPT-3.5(question + context)
else:
    Answer = GPT-3.5(question)
```

### Pipeline

```
User Question
     │
     ▼
SentenceTransformers (all-MiniLM-L6-v2)
     │  384-dim vector
     ▼
FAISS IndexFlatIP
     │  top-3 passages
     ▼
Prompt Builder (strategy-dependent)
     │  enriched prompt
     ▼
GPT-3.5-turbo (OpenAI API)
     │  short answer
     ▼
Exact Match + Token F1 evaluation
```

---

## Results

Evaluated on a **250-question balanced sample** (150 retrieval-needed + 100 parametric).

### Overall (n=250)

| Strategy | Exact Match | Token F1 |
|----------|-------------|----------|
| No Retrieval | 0.128 | 0.160 |
| Always Retrieval | 0.348 | 0.389 |
| Adaptive (Oracle) | 0.260 | 0.298 |
| **TA-ARE (Paper method)** | **0.348** | **0.411** |

### Retrieval-Needed Questions (n=150)

| Strategy | Exact Match | Token F1 |
|----------|-------------|----------|
| No Retrieval | 0.040 | 0.058 |
| Always Retrieval | 0.260 | 0.288 |
| Adaptive (Oracle) | 0.260 | 0.288 |
| **TA-ARE** | **0.093** | **0.136** |

### Parametric Questions (n=100)

| Strategy | Exact Match | Token F1 |
|----------|-------------|----------|
| No Retrieval | 0.260 | 0.314 |
| Always Retrieval | 0.480 | 0.541 |
| Adaptive (Oracle) | 0.260 | 0.314 |
| **TA-ARE** | **0.730** | **0.823** |

### Key Findings

- **Always Retrieval confirms the paper's core result** — on retrieval-needed questions, No Retrieval scored only 0.040 EM vs 0.260 for Always Retrieval (6.5× improvement)
- **TA-ARE matches Always Retrieval overall** (0.348 EM) with higher Token F1 (0.411 vs 0.389)
- **TA-ARE dominates on parametric questions** (0.730 vs 0.480) — correctly skipping retrieval preserves GPT-3.5's parametric knowledge
- **TA-ARE underperforms on retrieval-needed questions** (0.093 vs 0.260) — GPT-3.5 still misses some retrieval decisions, confirming the paper's finding that LLMs are imperfect judges of their own knowledge gaps

---

## Project Structure

```
RAG_Project/
│
├── RAG_NLP_Project.ipynb        ← Main notebook (run this)
│
├── src/
│   ├── data_loader.py           ← Load & sample RetrievalQA dataset
│   ├── retriever.py             ← FAISS index + retrieve() function
│   ├── llm.py                   ← GPT-3.5-turbo via OpenAI API
│   ├── strategies.py            ← 4 prompt builders including TA-ARE
│   ├── evaluation.py            ← EM, Token F1, all experiment runners
│   └── visualisation.py        ← All charts and diagrams
│
├── diagram/
│   ├── rag_pipeline.png                ← Our RAG pipeline diagram
│   ├── original_paper_architecture.png ← Paper architecture (3-part)
│   ├── results_chart.png               ← 3-strategy baseline chart
│   ├── results_chart_taare.png         ← 4-strategy comparison chart
│   └── retrieval_accuracy.png          ← TA-ARE retrieval accuracy chart
│
├── results.pkl                  ← Saved baseline experiment results
├── taare_results.pkl            ← Saved TA-ARE experiment results
└── README.md                    ← This file
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Nishi-013/RAG_Project
cd RAG_Project
```

### 2. Install dependencies
```bash
pip install datasets sentence-transformers faiss-cpu openai
```

### 3. Add your OpenAI API key
Open `src/llm.py` and replace line 22:
```python
OPENAI_API_KEY = "your-api-key-here"   # ← replace with your key
```
Get your key at: https://platform.openai.com/account/api-keys

### 4. Run the notebook
Open `RAG_NLP_Project.ipynb` in Jupyter and run:
```
Kernel → Restart & Run All
```

> **Note:** The full experiment takes ~20-30 minutes due to API calls.  
> Results are saved to `results.pkl` and `taare_results.pkl` so you don't need to rerun if the kernel restarts.

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥1.0 | GPT-3.5-turbo API |
| `datasets` | ≥2.0 | Load RetrievalQA from HuggingFace |
| `sentence-transformers` | ≥2.0 | Text embeddings (all-MiniLM-L6-v2) |
| `faiss-cpu` | ≥1.7 | Vector similarity search |
| `pandas` | ≥1.3 | Data manipulation |
| `numpy` | ≥1.21 | Numerical operations |
| `matplotlib` | ≥3.4 | Charts and diagrams |

---

## Differences from the Original Paper

| Aspect | Original Paper | Our Implementation |
|--------|---------------|-------------------|
| **Language Model** | GPT-3.5-turbo | GPT-3.5-turbo ✓ |
| **Adaptive Method** | TA-ARE prompting | TA-ARE implemented ✓ |
| **Retrieval System** | BM25 + dense retriever (live) | Pre-retrieved passages from dataset |
| **Questions Evaluated** | Full 2,785 questions | 250-question balanced sample |
| **LLMs Compared** | 6 models (TinyLlama to GPT-4) | Single LLM (GPT-3.5-turbo) |
| **Self-RAG comparison** | Yes — LLaMA-7B fine-tuned model | Not implemented (requires A100 GPU) |
| **Infrastructure** | Cloud API | OpenAI API ✓ |

---

## References

[1] Zhang, Z., Fang, M., & Chen, L. (2024). RetrievalQA: Assessing Adaptive Retrieval-Augmented Generation for Short-form Open-Domain Question Answering. *ACL Findings 2024*. https://arxiv.org/abs/2402.16457

[2] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

[3] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*. https://www.sbert.net

[4] Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*. https://github.com/facebookresearch/faiss

[5] Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *JMLR*.

[6] Wolf, T., et al. (2020). HuggingFace Transformers: State-of-the-Art Natural Language Processing. *EMNLP 2020*. https://huggingface.co/docs/transformers

[7] OpenAI. (2023). GPT-3.5-turbo. https://platform.openai.com/docs/models/gpt-3-5
