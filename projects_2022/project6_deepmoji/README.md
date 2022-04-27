# What You need for this Project
Download the repo as it is intended to be uploaded onto Google Drive and Mounted via Colaboratory Notebooks.
The files are setup so that requirements.txt will install all the necessary libraries, and the source code will be imported via the src/deepmoji directory. The helper functions are also located in src. The datasets used are contained within the data folder which will be where model training is done.

# Sources
A majority of the work is done from:

- Felbo, B., Mislove, A., Søgaard, A., Rahwan, I., &amp; Lehmann, S. (2017, October 7). Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm. arXiv.org. Retrieved April 19, 2022, from https://arxiv.org/abs/1708.00524 
- Devlin, J., Chang, M.-W., Lee, K., &amp; Toutanova, K. (2019, May 24). Bert: Pre-training of deep bidirectional Transformers for language understanding. arXiv.org. Retrieved April 19, 2022, from https://arxiv.org/abs/1810.04805 

By refactoring the Felbo et al., code, as well as converting from a Python 2.7 backend and Tensorflow v1 code base, the code was made to run on Python 3.X as well as Tensorflow v2 with Keras backend. The Transformer used in our report was obtained from the Hugging Face library.

# Usage
The most common issue when running the code was ensuring the directories lined properly with the code. Having dealt with those problems, the code runs seamlessly with a GPU runtime type on Colab. 

The notebook is setup for loading the pickle dataset, training the individual models, as well as evaluating the accuracy metrics.
