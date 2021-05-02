# Group10_NLP_Final_Project
Understanding and implementing research paper : ***Tweet2Vec: Character-Based Distributed Representations for Social Media***

# Contributors: 
Kumara Prasanna Jayaraju and Rinaldo Sonia Joseph Santhana Raj


# Background:
This repository provides a character-level encoder/trainer for social media posts. See [Tweet2Vec](https://arxiv.org/abs/1605.03481) paper for details. There are two models implemented in the paper - the character level tweet2vec and a word level baseline. 

# Data Collection:
Unfortunately, We didnt have any dataset associated with the paper due to confidentiality. So we planned to collect tweets on our own, based on the following common life oriented keywords such as '#life', '#motivation', '#happy', '#emotions', '#friends', '#babies', '#dogs' to implement and test out the project.

# Implementation:
We have divided the paper into two parts and you can find the notebook and supportive files in their respective diretories.
- Part 1: Comparing the correctly predicted hashtags by Word Model baseline with Tweet2Vec
- Part 2: Training and testing the dataset to calculate Precision, Recall and Mean Rank.

# Conclusion:
Our learning from this project is that, tweet2vec encoder performs better than word baseline for social media posts trained using supervision from associated hashtags. However, based on our observation there were few tweets were words baseline had better prediction of hastags as well. With respect to performance, without doubt tweet2vec outperforms the word baseline. This paper was limited to English language however the model can be extended to other languages as well. Future direction of the project will focus on how the model can be used for domains specific classification such as news feeds, social media and any content based platforms.

# Reference:
[1]: [Dhingra1 et al.2016] Bhuwan Dhingra1, Zhong Zhou2, Dylan Fitzpatrick1,2
Michael Muehl1 and William W. Cohen1, Tweet2Vec: Character-Based Distributed Representations for Social Media, Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers),2016

[2]: [Bengio et al.2003] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Janvin. 2003. A neu- ral probabilistic language model. The Journal of Machine Learning Research, 3:1137–1155.

[3]: [Godin et al.2013] Fréderic Godin, Viktor Slavkovikj, Wesley De Neve, Benjamin Schrauwen, and Rik Van de Walle. 2013. Using topic models for twit- ter hashtag recommendation. In Proceedings of the 22nd international conference on World Wide Web companion, pages 593–596. International World Wide Web Conferences Steering Committee.

[4]: [Zhangetal.2015] XiangZhang,JunboZhao,andYann LeCun. 2015. Character-level convolutional net- works for text classification. In Advances in Neural Information Processing Systems, pages 649–657.

GitHub (see data here):https://github.com/MaraPrazzy/Group10_NLP_Final_Project