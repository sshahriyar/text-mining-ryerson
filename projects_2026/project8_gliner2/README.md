# GLiNER2 for Performing Multiple NLP Tasks Efficiently

## Team Members

* Lok Prakash Pandey — [lok.pandey@torontomu.ca](mailto:lok.pandey@torontomu.ca)
* Samuel Ukoha — [samuel.ukoha@torontomu.ca](mailto:samuel.ukoha@torontomu.ca)
* Xinquan Yu — [xinquan.yu@torontomu.ca](mailto:xinquan.yu@torontomu.ca)

---

## Abstract

The Natural Language Processing (NLP) tasks such as Named Entity Recognition (NER), Text Classification (TC), Structured Data Extraction (SDE), and Relation Extraction (RE) are normally done using separate task-based models. This causes an increase in cost and complexity of managing such a system. In this respect, this project introduces the **GLiNER2** (Generalist Model for Named Entity Recognition 2) framework that can perform the mentioned tasks using a single unified model. This project reproduces the results done by the authors of GLiNER2 and evaluates its practicality for NLP tasks in a CPU-based environment. Overall, the implementation results showed that GLiNER2 offers a real-world unified solution for building efficient, flexible, and cost-effective NLP systems.

---

## Project Overview

This project focuses on reproducing the GLiNER2 model and evaluating its ability to perform multiple NLP tasks.

### Key Objectives

* Reproduce GLiNER2 model behavior
* Perform multi-task NLP (NER, TC, SDE, RE)
* Evaluate performance in a CPU-only environment

---

## Repository Structure

```
GLiNER2-Project/
│
├── README.md
├── GLiNER2_Project.ipynb
├── data/
├── src/
```

---

## Setup Instructions

Run the following commands before executing the notebook:

```python
!pip install -q gliner2
!apt-get -qq update
!apt-get -qq install git-lfs
!git lfs install
```

---

## How to Run

1. Open `GLiNER2_Project.ipynb` using Google Colab or Jupyter Notebook
2. Run all cells sequentially
3. Make sure that all required dependencies are installed before execution

---

## Dataset

* The project's notebook uses publicly available datasets compatible with GLiNER2
* Additional data (if required) can be downloaded within the notebook or from standard sources such as Hugging Face

---

## Results

* Successfully reproduced core functionality of GLiNER2
* Demonstrated multi-task capability within a single unified model
* Achieved efficient execution in a CPU-based environment
* Reduced dependency on multiple task-specific NLP models

---

## Limitations

* Performance is dependent on the quality of schema definitions
* Limited large-scale evaluation due to computational constraints

---

## Future Work

* Evaluate model performance on larger and more diverse datasets
* Compare against fine-tuned transformer-based baselines
* Extend implementation to real-time and production-level NLP systems

---

## Appendix 

**Comparison of Implementation**

| What we implemented | What we did not implement |
|---|---|
| Used the released pretrained GLiNER2 model | Did not train GLiNER2 from scratch |
| Implemented the main schema-driven usage idea of the paper | Did not rebuild the full training pipeline |
| Demonstrated text classification | Did not reproduce all classification benchmarks in the paper |
| Demonstrated named entity recognition | Did not reproduce the full NER benchmark setting from the paper |
| Demonstrated structured / JSON extraction | Did not systematically evaluate all structured extraction settings |
| Used the official GLiNER2 API and notebook-based implementation | Did not re-implement the internal architecture details |
| Showed how one model can support multiple tasks through schema design | Did not fully reproduce the paper as a research-level benchmark study |

---

## References

[1] Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois, *GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer*, arXiv, 2023

[2] Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, Ash Lewis, *GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface*, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2025, pp. 130–140

