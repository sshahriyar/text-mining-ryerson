# LLM Trees - Decision Trees through LLMs
![python](https://img.shields.io/badge/python-3.12.13%2B-blue)

## OVERVIEW

This repository is on the *replication* of the research paper: 

> #### "Oh LLM, I'm Asking Thee, Please Give Me a Decision Tree": Zero-Shot Decision Tree Induction and Embedding with Large Language Models (Knauer et al., 2025)

The original research paper can be found here: [link to original paper](https://dl.acm.org/doi/10.1145/3711896.3736818)  
Alternative link to research paper: [alt link](https://arxiv.org/abs/2409.18594)  

Original paper repository: [link to original repo](https://github.com/ml-lab-htw/llm-trees)  

<br>

The original paper implements the following:  

**1. Zero-shot Decision Tree Induction**  
- Prompting LLM(s) using zero-shot prompting to derive a decision tree classifier by *only* passing features of the dataset (no training data passed to LLM)  

**2. Embedding Induction**  
- Generate decision trees whose nodes are used to create binary feature vectors as inputs for neural network models  
  
<br> 

**Motivation**
The *motivation* behind the paper comes from **knowledge distillation** of LLMs (their "world knowledge") and the idea that data can be scarce or propriety. This paper also draws from **in-context learning** and **transfer learning**. The key idea is to find a way to use the "world knowledge" that large language models have, which have been trained on an enormous amount of data from the internet, and derive *models* (in this case, decision trees), **without any training data** ever passed to the LLM. This overcomes the issue of privacy concerns with proprietary data and demonstrates how LLMs can still be leveraged for small datasets that would otherwise be difficult to train in models.  

> Exercept from the paper: "we present the first approach to apply state-of-the-art LLMs for zero-shot model generation using in-context learning, i.e., we show how LLMs can build intrinsically interpretable trees without access to pretrained model weights and without any training data" (Knauer et al., 2025)  

<u>Domains of study:</u> large language models, zero-shot prompting, in-context learning, knowledge distillation, transfer learning, data scarcity, intrinsic model induction   

<br> 

## OUR PROJECT  
**Our project replicates the original research paper by implementing <u>both</u> (1) Decision Tree Induction and (2) Embedding Induction.  
However, we use <u>different LLM models and only 5 datasets</u> in comparison.**  

Datasets Used:  
1. bankruptcy.py
2. boxing1.py
3. boxing2.py
4. colic.py
5. creditscore.py  

LLM Models Used (Ollama):  
1. gemma3
2. gpt_oss
3. mistral_small3
4. qwen3  

Part 1 Baseline Models Used: Autogluon, TabPFN  

  
Part 2 Baseline Models Used: 
* LLM Embeddings
* Baseline - Uses the raw input values
* RandomTreeEmbeddings
* RandomTrees - Supervised
* RandomTrees - SemiSupervised
* ExtraTrees - Supervised
* ExtraTrees - Semisupervised
* XGBoost - Supervised
* XGBoost - Semisupervised  


<br>  

**Note:** You can find a table of implementation comparison in *LLM_trees_project_report.ipynb* to see how our implementation differs/matches the original paper  
  

## Required Environment
- python >= 3.12.13
``` python
pip install -r requirements.txt
```
* You will also need to install Ollama software from, https://ollama.com/download/windows
* In the env terminal after running requirements.txt, which also installs ollama python package to env, you will need to run the following commands to pull the models to your env.
```bash
ollama pull gpt-oss:20b
ollama pull qwen3:14B
ollama pull gemma3:12b
ollama pull mistral-small3.2:24b
```
  
<br>  

## Required Dependencies  
**Please refer to the `requirements.txt` file** for required dependencies as some models require API access. 

## Part 1: Decision Tree Induction  
**Follow along the project report notebook for the entire pipeline. Make sure to run each cell from the start.**  

Only the Ollama LLMs are prompted multiple times to generate decision trees transformed into python functions. The prompted decision tree functions can be found in `data/llm_induction/model_name/dataset_name/dt_func_#.txt` where model_name is one of above llms and dataset_name is one of the 5 datasets. Each dataset has 5 functions, and is prompted for each of the 4 llms.  
You can find the prompt in `src/pompter.py` as a reference if you are creating your own prompts.  

### Evaluate LLM Trees
This is done using the [project report notebook](LLM_Trees_project_report.ipynb) which you can easily follow along to test the functions that we extracted under the section: Induction Evaluation. 


## Part 2: Embedding Induction  
The decision tree functions as a result of prompting the llms can be found in `data/llm_embeddings/model_name/dataset_name/dt_func_#.txt`.  
Refer to *Step 2* in `src/prompter.py` for help with prompting for embeddings (if prompting yourself).  

### Evaluate LLM Embeddings  
Again you may follow the [project report notebook](LLM_Trees_project_report.ipynb) under the section: Embedding Evaluation. 


## SUMMARY
Replication of Oh LLM, I’m Asking Thee, Please Give Me a Decision Tree”: Zero-Shot Decision Tree Induction and Embedding with Large Language Models (KDD conference) paper for DS8008 Class.
- We prompt 4 Open Source Models from Ollama with informationa about a dataset, such as features, and ask it to generate decision tree logic internally.
- It also then takes that internal logic and creates a function that can be used as a classifier.
- The logic of these functions are then used to evaluate the performance of of these decision trees in classification and embedding extraction.
- Each dataset is used to prompt the LLM 5 times, to generate 5 different decision trees. This is done twice, once for decision tree induction method, and another time for decision tree embedding method.
