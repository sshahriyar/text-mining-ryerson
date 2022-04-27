# DeepTriage Implementation
Partial Implementation of paper "Deeptriage: Exploring the Effectiveness of Deep Learning for Bug Triaging" for Ryerson University, DS8008 Natural Language Processing W2022 Course as part of the Ryerson Data Science Master's (MSc) program. 

# Data
The bug data used is sourced from a Kaggle notebook [here](https://www.kaggle.com/datasets/crawford/deeptriage). The datasets 'deep_data.csv' and 'classifier_data_0.csv' are used in this implementation project. 

# Setup 
After downloading the dataset provided above, ensure that it is placed inside the 'data/' folder. 
To run models, simply follow the cells in 'Final_project_DeepTriage.ipynb', but be sure to adjust any function arguments labelled 'overwrite' to True (these are programmed to run any processing and modelling steps from scratch. When False, these functions simply read pre-existing files). If starting the notebook from scratch, change 'overwrite' to 'True'. 

# Reference Paper
[Deeptriage: Exploring the Effectiveness of Deep Learning for Bug Triaging](http://bugtriage.mybluemix.net/)