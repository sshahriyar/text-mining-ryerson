#  LLM2Rec-based Recommendation System

> The implementation of the LLM2Rec model using CSFT IEM and SASRec for sequential recommendation, and a baseline model using BERT embeddings and SASRec for sequential recommendation.
---

## 📌 Overview

This project focuses on Sequential Recommendation, where the goal is:

Predict the next item a user will interact with based on past behavior.

We implement:

- Baseline Model: BERT + SASRec  
- Advanced Model: LLM2Rec + SASRec framework  

---

## 🗂️ Project Structure
```
NLP_project_repo/
├── data/                      # Dataset
├── src/                       # Modular source code
│   ├── __init__.py
│   ├── data_loader_baseline.py
│   ├── embeddings_baseline.py
│   ├── model_baseline.py
│   ├── dataset_baseline.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   ├── inference_baseline.py
│   ├── utils_baseline.py
│   ├── csft.py
│   ├── data_utils.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── mntp_contrastive.py
│   └── sasrec.py
├── Final_project_Report_Group_05.ipynb   # Report notebook
├── baseline_model.py                    # Baseline pipeline
├── final_report.py                      # Full LLM2Rec pipeline
└── README.md
```
---

## 📊 Dataset

Dataset: Amazon Movies & TV (5-core)

Dataset Source: https://www.kaggle.com/datasets/wajahat1064/amazon-reviews-data-2023/data

Contains:
- User interaction sequences  
- Item titles (metadata)  
- Train / validation / test splits  
---

## 🧠 Models

### 🔹 1. Baseline Model — BERT + SASRec

File: baseline_model.py

Components:

- BERT → converts titles into embeddings  
- Adapter layer → projects 768 → 128  
- SASRec → sequence modeling  

Pipeline:

Item Titles → BERT → Embeddings → Adapter → Transformer → Prediction  

---

### 🔹 2. Advanced Model — LLM2Rec + SASRec

File: final_report.py

Includes:

- CSFT (Collaborative Supervised Fine-Tuning)  
- MNTP (Masked Next Token Prediction)  
- Contrastive Learning  
- Embedding Extraction  
- SASRec Evaluation  

Pipeline:

User Sequences + Item Titles → CSFT → MNTP → Contrastive Learning → LLM2Rec Item Embeddings → SASRec → Prediction

---
## ⚙️ Environment Setup
```
Tested on Google Colab (A100 GPU) and local Python 3.12.13 environment.
```
---
## ⚙️ Installation
```
pip install transformers torch numpy pandas tqdm  
```
---

## ▶️ How to Run (Google Colab)

Step 1: Upload dataset to Google Drive

Step 2: Mount Drive  

```
from google.colab import drive  
drive.mount('/content/drive')  
```
Step 3: Extract Dataset  

```
!unzip -q "/content/drive/MyDrive/data/Movies_and_TV_data.zip" -d "/content/data"
```

Step 3: Run these .py files .ipynb as notebooks

- baseline_model.py  
- final_report.py  

---

## 📈 Evaluation Metrics

- Recall@K  
- NDCG@K  

---

## 🎯 Example Inference

user_history = ["Avengers", "Ironman"]  

Model predicts next likely items.

---

## 👥 Team

- Jakia Nowshin- 501405984  
- Ishrat Jaben Bushra- 501338510  
- Surayia Rahman- 501145340 
