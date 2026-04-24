# Evaluating LLMs on Stable Matching and Preference Reasoning Tasks

## Authors
Dana Aljamal  
Shahida Batool  
Nimrah Adam  

---

## Overview

This project evaluates how large language models (LLMs) perform on structured reasoning tasks based on the Stable Matching Problem. It directly implements the experimental framework proposed in the paper *"Matching Markets Meet LLMs: Algorithmic Reasoning with Ranked Preferences"* (Hosseini et al., 2025).

We explicitly evaluate how different levels of model scaling impact performance locally and globally. All baseline and reasoning inferences are streamed via the **Groq API** for high-speed resolution.

We compare three primary models:
1. **Basic Model (`llama-3.1-8b-instant`)**: Evaluates foundational instruction following.
2. **Reasoning Model (`llama-3.3-70b-versatile`)**: Assesses if structural parameter scaling effectively improves global stability reasoning.
3. **Advanced Scaling Model (`Gemini`)**: Used exclusively to evaluate structural performance collapse when market sizes scale exponentially (see `gemini_stable_matching_evaluation.ipynb`).

The goal is to assess:

- Correctness and validity of model-generated matchings  
- Stability of matchings (absence of blocking pairs)  
- Reasoning ability across tasks of increasing complexity  
- Performance differences between the basic and reasoning models  

---

## References

- Hadi Hosseini, Samarth Khanna, Ronak Singh (2025). *Matching Markets Meet LLMs: Algorithmic Reasoning with Ranked Preferences*. arXiv:2506.04478. [Link to paper](https://arxiv.org/pdf/2506.04478)

---

## Dataset

All primary experiments use synthetically generated instances from the **Impartial Culture (IC)** setting, where agents hold uniform, truly random preference distributions.

| File | Description | Used In |
| :--- | :--- | :--- |
| `5_ic_processed.csv` | 5 Proposers × 5 Acceptors (10 agents total) | All Tasks 1–4 in the main notebook (`llm_stable_matching_evaluation.ipynb`). Evaluated on 10 and 20 sampled instances. |
| `20_ic_processed.csv` | 10 Proposers × 10 Acceptors (20 agents total) | Gemini scaling evaluation only (`gemini_stable_matching_evaluation.ipynb`). Not included in this repository. |
| `50_ic_processed.csv` | 25 Proposers × 25 Acceptors (50 agents total) | Gemini scaling evaluation only (`gemini_stable_matching_evaluation.ipynb`). Not included in this repository. |

> **Note:** The `20_ic_processed.csv` and `50_ic_processed.csv` files were used externally to test how Gemini performance degrades as market size increases. However, `50_ic_processed.csv`  was not included in this repository due to file size.

---

## Tasks

### Task 1 — Generate Stable Matching

Given preference lists for all agents, the model must generate a complete and **stable** one-to-one matching.

- **Input**: Preference lists for both groups  
- **Output**: One-to-one matching in JSON format  

**Metrics:**
- Parsed count: Model successfully output readable JSON.
- Valid matching: Model generated a strict one-to-one matching (no duplicates).
- Stable matching: The matching contains zero blocking pairs (globally stable).
- Exact match with ground truth  
- Average blocking pairs  

*(Execution Warning: High token cost and latency due to generative combinatorial complexity)*

---

### Task 2 — Detect Instability

Given a matching and preference lists, the model must determine whether the matching is stable.

- **Input**: A matching along with preference lists  
- **Output**: YES (stable) or NO (unstable)  

**Metrics:**
- Correct predictions: The model accurately identified if the matching contained blocking pairs.
- Accuracy  

---

### Task 3 — Resolve Instability

Given an **unstable** matching, the model must repair it into a stable one while maintaining a valid one-to-one assignment.

- **Input**: An unstable matching along with preference lists  
- **Output**: A stable one-to-one matching in JSON format  

**Metrics:**
- Stable matching count: The model successfully repaired the broken matching to achieve global stability.
- Exact match count  
- Average blocking pairs  

*(Execution Warning: High token cost and latency due to complex structural resolution requirements)*

---

### Task 4 — Preference Reasoning

The model answers granular preference-based reasoning queries derived from ranked lists.

- **Input**: Preference lists and a query about agent preferences  
- **Output**: A correct agent identifier derived from preference rankings  

**Metrics:**
- Correct answers: The model logically understood granular preference rankings between two agents.
- Accuracy  

---

## Key Results and Insights

| Task | Basic Model | Reasoning Model |
| :--- | :--- | :--- |
| Task 1: Stable Matching Generation | 6 / 20 stable (30%) | 5 / 20 stable (25%) |
| Task 2: Instability Detection | 50% accuracy | 60% accuracy |
| Task 3: Instability Resolution | 20% stable | 20% stable |
| Task 4: Preference Reasoning | 55% accuracy | 70% accuracy |

**Key observations:**
- Models achieve high validity but struggle with stability, indicating difficulty in enforcing global constraints  
- Stable matching generation performance is low (~30%), showing challenges in multi-step reasoning  
- Preference reasoning performs best (up to ~70% accuracy), highlighting strength in local reasoning  
- The reasoning model does not consistently outperform the basic model, indicating reliability limitations  
- Gemini performs well on smaller datasets but degrades significantly on larger instances, highlighting scalability challenges  

---

## Setup & Running the Notebook

### Requirements

Install dependencies via pip:

```bash
pip install langchain-groq pandas matplotlib numpy
```

### API Key

A **Groq API key** is required. When you run the notebook, you will be prompted to enter it. You can get a free key at [console.groq.com](https://console.groq.com).

### How to Run

1. Open `notebooks/llm_stable_matching_evaluation.ipynb` in Jupyter or VS Code.
2. Run cells sequentially from top to bottom.
3. When prompted, enter your Groq API key.
4. Results are automatically saved to `results/summaries/` as JSON files.
5. The final bar chart is saved to `results/figures/`.

> **Note:** Task 1 and Task 3 cells are the most computationally expensive and will take longer to complete due to the complexity of generating and repairing full stable matchings.

---

## Project Structure

```text
NLP/
├── data/
│   └── 5_ic_processed.csv          # Primary dataset used in all Tasks 1–4
│   └── 20_ic_processed.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # API key setup and model initialization
│   ├── data_utils.py               # Dataset loading and preprocessing
│   ├── io_utils.py                 # JSON summary save/load helpers
│   ├── reporting.py                # Visualization and chart generation
│   ├── task1_stable_matching.py    # Task 1 inference pipeline
│   ├── task2_detect_instability.py # Task 2 inference pipeline
│   ├── task3_resolve_instability.py# Task 3 inference pipeline
│   └── task4_preference_reasoning.py# Task 4 inference pipeline
│
├── notebooks/
│   ├── llm_matching_markets_evaluation.ipynb      # Main evaluation notebook (Tasks 1–4)
│   ├── gemini_stable_matching_evaluation.ipynb   # Gemini scaling evaluation
│   └── architecture.png                          # Architecture diagram (used in notebook)
│
├── results/
│   ├── summaries/                  # JSON output from each task run
│   ├── tables/                     # Formatted result tables
│   └── figures/                    # Generated bar chart and visualizations
│
└── README.md
```
