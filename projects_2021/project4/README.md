# NLP_Project
RMDL and RobertaBERT for Text Classification.
*********************************************************************************************************************************************************************************
## Paper Title and Link :
RMDL: Random Multimodel Deep Learning for Classification
github link : [https://github.com/kk7nc/RMDL](link)

## Description of Paper
*********************************************************************************************************************************************************************************
RMDL model can be seen as ensemble approach for deep learning models.RMDL solves the problem of finding the best deep learning structure and architecture while simultaneously improving robustness and accuracy through ensembles of deep learning architectures.

## Context of the Problem
********************************************************************************************************************************************************************************
* The continually increasing number of complex datasets each year necessitates ever improving machine learning methods for robust and accurate categorization of these data.
* Generally, deep learning models involves a lot of randomization
* Users need to manually do hyper parameter tuning by changing each and every parameter which results into longer execution times
* So, They proposed an ensemble based approach for deep learning models.

## Methodology
*********************************************************************************************************************************************************************************
![alt text](https://github.com/garima2751/NLP_Project/blob/main/images/rmdl_archi.png)

## Implementation 
*********************************************************************************************************************************************************************************
* To implement RMDL model please follow the code here [a link](https://github.com/garima2751/NLP_Project/blob/main/Src/stackoverflowRMDL.ipynb)
* please install RMDL library using !pip install RMDL command
* for Roberta BERT model implementations ! pip install simpletransformers and please follow the code here 
[a link](https://github.com/garima2751/NLP_Project/blob/main/Src/Roberta_stack.ipynb)

## Results and Conclusion
*********************************************************************************************************************************************************************************
<img src="https://github.com/garima2751/NLP_Project/blob/main/images/nlp_result_table.png" alt="drawing" width="500"/>

* In case of stack overflow dataset : BERT model outperformes RMDL models with an f1-score of 80 %
* In case of IMDB dataset : Both models perform equally well
* In case of Reuters dataset : RMDL perform slightly better than BERT 
