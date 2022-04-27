## NLP - Weak Supervision

Understanding & implementing *skweak: Weak Supervision by Easy for NLP* 
https://arxiv.org/pdf/2104.09683v1.pdf

****
**Requirements**
* To run **Weak_Supervision_Final_Project.ipynb** File, 
     * Download the data files named reuters_small.tar.gz & crunchbase_companies.json.gz from https://github.com/NorskRegnesentral/skweak/tree/main/data
     * Upload to Google Colab & run the file.
* In order to run the **NER_step_by_step.ipynb** File,
     * Download the following datafiles from https://github.com/NorskRegnesentral/skweak/tree/main/data: 
          * crunchbase_companies.json.gz
          * firstnames.json
          * form_frequencies.json
          * geonames.json
          * products.json
          * reuters_small.tar.gz
          * wikidata_small_tokenised.json.gz
     * Also download reuters_small.spacy from https://drive.google.com/drive/folders/1RTSxZZXfHZH1Cmx_WST8YWGbmyzmyBWw?usp=sharing
     * Upload all files to Google Colab & run the file.
****

**Problem Description:**

In NLP, accurately labeled data is scarce when dealing with specialized domains and internal business initiatives. NLP projects undertaken by businesses often deal with the lack of labeled data – especially when the business defines domain specific (e.g. internal) labels and cannot make use of pre-existing resources. In many cases, there is a need to rely on massive sets of hand-labeled domain-specific training data or large pre-trained language models.

**Context of the Problem:**

In the modern world, there is infinite availability of text document. Unfortunately, there is huge scarcity of meaningful labeled data. This issue magnifies in resource-poor languages and/or uncommon textual domains, also extended to projects without pre-existing datasets.This problem is important as Machine Learning Models require labeled data for supervised learning, and the scarcity of acurately labeled data makes it difficult to train these models.

**Limitation About other Approaches:**

There are many ways to get more labeled training data however each have their own drawbacks.

* **Traditional Supervision (Hand-Labeled Data by Subject Matter Experts (SMEs))**

     Hand-labeled training datasets are expensive and time-consuming to create, and are not able to swiftly accommodate to change if new labelling guidelines are introduced. For example, if a new domain-specific label was introduced, a full review of the training data set would be required by SMEs.

* **Semi-supervised Learning (Use of structural assumptions on unlabeled data)**

     The semi-supervised learning approach takes a small labelled dataset and a large unlabelled dataset to extract structural assumptions. However, it needs some labeled data.

* **Transfer Learning (Use of pre-trained models)**

     Transfer Learning makes use of already existing pre-trained models and fine-tunes them on a different task. However, transfer learning only works if the initial and target problems are similar enough for the first round of training to be relevant. This is often not the case in cross sector NLP projects.

**Solution:**

The solution to these problems is provided by the weak supervision paradigm. Weak Supervision saves the time that is required to label data by hand. It takes, as input, noisy/conflicting/less accurate data, then aggregates the data to provide a single output of labels without conflicts. This aggeregated data can then be used to train an NLP model for sentiment analysis and named entity recognition. The skweak framework relies on weak supervision to programmatically label data points through a collection of labelling functions which are discussed further in this report.
