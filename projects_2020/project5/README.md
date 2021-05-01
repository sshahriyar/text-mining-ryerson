# DS8008---Final-Project

<b>Data</b> Folder: Contains the 2 test/train news and medical dataset

<b>Files to run few-shot learning</b> Folder: Contains python modules that were created to run the few-shot learning code. These modules were obtained from the paper "Few Shot Text Classification with a Human in the Loop"

The project code can be found in the notebook "<b>DS8008_FinalProject_Group5.ipynb</b>"

<b>Project Description:</b><br>

Our project uses Active Learning and Few-shot learning for text classification. We tested our models on a news dataset and a medical dataset.

In certain cases, document labeling can be considered an expensive task. It requires an expert to read and manually determine the associated class labels. This process can be improved by using a prediction model trained over a set of documents with known labels, and using this model to predict the labels for the unlabeled documents. However, there remains a question of how big of a training set (and hence the total amount of manual labeling) is needed to obtain a prediction model with relatively good performance


<b>Active Learning (AL)</b> is an effective approach to speed up the prediction model construction through careful selection of the instances to be labeled and added to the training set. In this project, we consider various AL approaches, including disagreement sampling and uncertainty sampling for binary text classification over a publicly available dataset. Our numerical results obtained through employing Random Forest Classifier and Multinomial Naive Bayes classifier as the base prediction models indicate that AL strategies speed up the document labeling process.

<b>Few Shot Learning</b> is another approach to handling data with limited labels. With this approach we are able to classify an entire corpus of unlabeled documents using a "human in the loop" strategy where the content owner manually classifies just one or two duments per category and the rest is automatically classified. The key to this approach is the selection of the proper class representatives. For the rest of the documents we can use a similarly or distance score to assign them to their respective classes.
