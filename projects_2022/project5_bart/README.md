# DS8008_NLP_PROJECT_BART

This is a review of the paper "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension". 

**Link to paper:** https://arxiv.org/abs/1910.13461

**Abstract:**

The emergence of pre-trained models has revolutionized the applications of natural language processing. However, most of the pre-trained models do not perform well across different tasks limiting their applicability. The paper in review introduces BART as a Neural Network model which combines both Bidirectional and Auto-Regressive Transformers and pre-trains a model using a combination of noising schemes. It can be seen as generalizing many pretraining schemes and achieving noising flexibility.

**Report:**
The **Final_project_Group_4.ipynb** file contains the paper review along with implementation details.

**Source Code:**

The original implementation of the paper is in the initial commit at fairseq github repository at https://github.com/pytorch/fairseq/commit/a92bcdad5a0dea6a440cc92976e4166811b16671. 
Since then, there have been many improvements to the model at fairseq github repository: https://github.com/pytorch/fairseq/tree/main/fairseq/models/bart. 
However, the Huggingface implementation at https://github.com/huggingface/transformers/tree/main/src/transformers/models/bart is the more standard and reference in the project. Huggingface contains transformer library code which BART implementation was able to base on.
The src folder contains Huggingface BART model code.

**Dataset:**

Huggingface BART model has been trained on multiple datasets. Following are the links to the different vocabularies.

- "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/vocab.json"
- "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/vocab.json"
- "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/vocab.json"
- "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/vocab.json"
- "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/vocab.json"
- "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/vocab.json"

In addition to the datasets used in the paper, a custom dataset was also used in the project to compare BART and BERT base models. The custom dataset used in this experiment is a dataset downloadable from https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data and pre-processed. The pre-processed dataset is in the data folder.
