# Puns_Generation

This repository is a exploratory summary of the paper "Pun Generation With Surprise"

link to the paper: https://arxiv.org/abs/1904.06828

He He, Nanyun Peng, & Percy Liang (2019). Pun Generation with Surprise. In North American Chapter of the Association for Computational Linguistics (NAACL).

## Included Files
Pun_Generation.ipynb
  - The main report
  - Contains the code to run the pun generation system
utility.py
  - Functions required for the code in 'Pun_Generation.ipynb' to run

requirements.txt 
environment.yml
  - Specification regarding the environment required to run the system

## Required Environment 
Running this notebook requires a very specific setup, including Python version 3.6, and PyTorch version 0.4.0. Here is the information for setting up the environment properly.
First, navigate to the downloaded pungen folder.
If you are using Anaconda, run the following command (where envname is your environment name):

conda env create --name envname --file=environments.yml

If you are using venv, you can install packages via pip. After creating a new venv, use:
pip install -r requirements.txt

This will install the needed packages.

### fairseq Installation
When installing the package fairseq, note that the distribution being installed is not the most up to date version. The paper's original authors actually edited modules in fairseq, and have packaged their version on their github. While the requirements.txt file points to this distribution, to install the followings commands may be used.

git clone -b pungen https://github.com/hhexiy/fairseq.git
cd fairseq
pip install -r requirements.txt
python setup.py build develop
Directory structure/pretrained models

## Required Data

To ensure the directory structure fits for the program implementation and so avoid having to train very large models it is recommended you download our premade folder. The Wikitext model below will have to be aquired separately. https://drive.google.com/a/ryerson.ca/file/d/1Wmh7gbgxZV6GEPEl5F4Net8d4cHE3H7b/view?usp=sharing

### Wikitext Model

One of the required models, Wikitext-103, comes from the actual fairseq site. To setup properly, use these commands:

curl --create-dirs --output models/wikitext/model https://dl.fbaipublicfiles.com/fairseq/models/wiki103_fconv_lm.tar.bz2
tar xjf models/wikitext/model -C models/wikitext
rm models/wikitext/model
