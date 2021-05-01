# Semantic Relations between Wikipedia Articles

<a href="https://colab.research.google.com/github/masukislam/DS8008_NLP/blob/master/demoRun.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


Implementation, trained models and result data for the paper **Pairwise Multi-Class Document Classification for Semantic Relations between Wikipedia Articles** [(PDF on Arxiv)](https://arxiv.org/abs/2003.09881). 
The supplemental material is available for download under [GitHub Releases](https://github.com/masukislam/DS8008_NLP/Releases) or [Zenodo](https://doi.org/10.5281/zenodo.3713183).

![Wikipedia Relations](https://github.com/masukislam/DS8008_NLP/raw/master/wikipedia_example.png)



# Wikipedia corpus
wget https://github.com/masukislam/DS8008_NLP/releases/download/Models/enwiki-20191101-pages-articles.weighted.10k.jsonl.bz2
bzip2 enwiki-20191101-pages-articles.weighted.10k.jsonl.bz2

# Train and test data
wget https://github.com/masukislam/DS8008_NLP/releases/download/Model/train_testdata__4folds.tar.gz
tar -xzf train_testdata__4folds.tar.gz

# Models
wget https://github.com/masukislam/DS8008_NLP/releases/download/Model/model_wiki.bert_base__joint__seq512.tar.gz
tar -xzf model_wiki.bert_base__joint__seq512.tar.gz
```


## Demo

You can run a Jupyter notebook on Google Colab:

<a href="https://github.com/masukislam/DS8008_NLP/blob/master/demoRun.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

