# NLPProject

**Title:**<br />
 *Evaluating Doc2Vec Model based on Sentiment Analysis on IMDB DataSet (Using Gensim)*
 
 **Instruction** <br />
 The total end-to-end project including loading the data is contained in the Jupyter Notebook. 
 You need to install gensim ,gensim.utils packages
 
 
 **Background Information** <br />
 Many ML algorithms require the input to be presented as a fixed-length feature vector. (For example Bag-Of-Words is one of the most common fixed-length feature.)
Bag-Of-Words has two major weaknesses: 
1)Loses the  ordering of words 2)Ignores semantics of words

Doc2Vec (Paragraph Vector) is an unsupervised algorithm that learns fixed-length feature representations from variable-length pieces of text.
 In this algorithm each document is represented by a dense vector which is trained by predicting words in the document. 
This method overcomes the Bag-Of-Words weaknesses.


**Word2Vec Vs. Doc2Vec** <br />
In Word2Vec, you train to find word vectors and then you use these word vectors in NLP tasks.
Doc2Vec generates vectors representing documents (sentences, paragraphs) 
While Word2Vec computes a feature vector for every word in the corpus, Doc2Vec computes a feature vector for every document in the corpus. 

**How Does it Work?** <br />
Both Word2Vec and Doc2Vec train models to predict one or more words. The weight matrices that are created during this training are the word and document vectors.
Gensim allows you to train Doc2Vec with or without word vectors

**Paragraph Vector : A distributed memory model (PV-DM)** (Right Image)<br />
The basic idea behind PV-DM is inspired from Word2Vec . Every paragraph is mapped to a unique vector represented by a column in matrix D  and every word is mapped to a unique vector represented by a column in matrix W. Paragraph vector or word vectors are averaged or concatenated to predict the next word in context.
After being trained , the paragraph vector can be used as a feature for the paragraph. 

**Paragraph Vector without ordering :Distributed bag of words (PV-DBOW)** (Left Image) <br />
In this method we ignore the context words in input but force the model to predict words randomly sampled from the paragraph.In this version, the paragraph vector is trained to predict the words in small windows.  


<img src="https://user-images.githubusercontent.com/81987771/115461302-43b71c80-a1f7-11eb-8ec8-17ebb5422f6f.png" width="500"/> <img src="https://user-images.githubusercontent.com/81987771/115461659-adcfc180-a1f7-11eb-8326-f24928e7c1f2.png" width="500"/> 


**Dataset:**<br />
We used the imdb Dataset from below link here: ( It will be downloaded as one of the steps in the Jupyter notebook file)
<http://ai.stanford.edu/~amaas/data/sentiment/>`_ 

These reviews will be the documents that we will work with. There are 100 thousand reviews in total.
25k reviews for training (12.5k positive, 12.5k negative)
25k reviews for testing (12.5k positive, 12.5k negative)
50k unlabeled reviews


**Methodologies:**<br />
Our starting point was to replicate part of the papers listed below which includes the original papers on Doc2Vec concept.
We chose to test the application of Doc2Vec on sentiment analysis.
The authors did not publish their code .However there were several implementations of their papers. We chose one of those implementations as a baseline.
That implemenation trained the models based on Gensim(Doc2Vec) and then assess the sentiment using  Logistic Regression. We expanded that assessment of the sentiment using RandomForest Classifier and GaussianNB.<br />
*Other papers mentioned the difficulty in replicating the original papers both for accuracy and in terms of the best models and hyperparameters.We performed several experiments to determine the best Doc2Vec model in predicting sentiment.*

Tuning Models were built using DBOW, DM(with averaging the word vectors), DM with concatenating the word vectors as well as combinations of DBOW and each of the two DM models.

As well, different window sizes were tested between 5 and 10. The initial learning rate, alpha, was tried between 0.01 and 0.05. And the final learning rate was tried between 0.0001 and the starting rate.

The sample parameter was tried for how much higher frequency words are down sampled and was set at 0 in the end.

The negative sampling (with 5 words) was tried based on literature.

The final models chosen were: DBOW (due to itès consistently low error rate with the advantage of less memory required) window size of 5 Alpha of 0.025 final Learning rate of 0.0001 <br />

**Results:**<br />
The Doc2Vec is a strong methodology for predicting sentiments.However the model has to be chosen carefully and it has to be tuned.
The original papers indicated that a more complex concatenated models had the lowest error rate however we found that the simpler DBOW Model was the same or better.

![Screen Shot 2021-04-20 at 7 45 29 PM](https://user-images.githubusercontent.com/81987771/115480324-41b18580-a218-11eb-9bcf-652fc585ab9a.png)


**References:**<br /> 
We try to replicate the concepts of these two papers:<br />
* Paper 1:* https://arxiv.org/abs/1405.4053       * Paper 2:* https://arxiv.org/abs/1507.07998

Gensim – Deep learning with paragraph2vec (https://radimrehurek.com/gensim/models/doc2vec.html)

https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html#sphx-glr-download-auto-examples-howtos-run-doc2vec-imdb-py





