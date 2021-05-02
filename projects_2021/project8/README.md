# Code for Knowledge-guided Open Attribute Value Extraction with Reinforcement Learning 
## Requirements:

ray[debug]==0.7.5

python-Levenshtein

redis

ray

ray[rllib

unicodedata2

tensorflow==1.15.1

# Final Results

| MODEL           | GPU        | GAME  |MOVIE | PHONE |
| -------------   |:----------:| -----:|-----:|-----: |
| Random(BiDAF)   | 0.259      | 0.155 |0.267|0.223|
| First(BiDAF)    | 0.451      | 0.498 |0.533|0.632|
| Majority(BiDAF) | 0.488      | 0.321 |0.539|0.415|
|Confidence(BiDAF)|**0.799**    |0.488  |0.645|0.560|
|RL-KG(BiDAF)     |    0.669   |  **0.561** |             **0.648**     |          **0.611** |
Oracle(BiDAF)       |    0.902  |       0.793   |       0.846     |      0.812   |




| MODEL           | GPU        | GAME  |MOVIE | PHONE |
| -------------   |:----------:| -----:|-----:|-----: |
Random(BERT)     | 0.374 | 0.234 | 0.361 | 0.287 |
First(BERT)      | 0.581 | 0.579 | 0.670 | 0.738| 
Majority(BERT)    |0.502  |0.394 | 0.625 | 0.449 |
Confidence(BERT)  |0.727 | 0.600  |0.552 | 0.540 |
RL-KG(BERT)(Pretrained)     |**0.744**    |**0.681**   | **0.758**   |   **0.808** |   
RL-KG(BERT)(Ours) | 0.534|0.603| 0.724|0.702|
Oracle(BERT)     | 0.925 | 0.857 | 0.887 | 0.909|


| MODEL           | GPU        | GAME  |MOVIE | PHONE |
| -------------   |:----------:| -----:|-----:|-----: |
Random(QANet)     | 0.261   |      0.167  |        0.259 |          0.236  |        
First(QANet)       |0.507    |     0.533   |       0.531  |         0.675   |       
Majority(QANet)    |0.484     |    0.325    |      0.500   |        0.469    |      
Confidence(QANet) | 0.691      |   0.493     |     0.689    |       0.546     |                                                                  
RL-KG(QANet)      |  **0.699**    |       **0.673**|            **0.730**|                  **0.712**     |     
Oracle(QANet)      |0.932        | 0.840       |   0.878          | 0.868  |





Liu, Ye, Sheng Zhang, Rui Song, Suo Feng, and Yanghua Xiao. “Knowledge-Guided Open Attribute Value Extraction with Reinforcement Learning.” ArXiv:2010.09189 [Cs], October 18, 2020. http://arxiv.org/abs/2010.09189.

