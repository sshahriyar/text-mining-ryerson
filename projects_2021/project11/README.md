# Seq2SeqChatbotWithAttention

#### Problem Description:

Previously, chatbots have been developed using hand-written rules, making models static. With the recent focus in the area of deep learning and natural language processing (NLP), there have been several advancements in the space. For example, Sequence to Sequence (seq2seq) models use recurrent neural networks to solve complex language problems. The general architecture consists of an encoder and a decoder, each being stacks of LSTM or GRU layers. But, Seq2Seq models lack the ability to work with larger sentences because all information from the input sentence is required to be encoded into a fixed length vector, and is sequentially passed until the output is produced. Therefore, knowledge of the input is slowly lost throughout the prediction process.

#### Solution:

To improve the seq2seq model and make it more robust, during training, an attention mechanism is included to gather context for each input word, allowing the decoder to have more information about each part of the input sentence. To generate the next word in the output, the context from the user input and the generated output so far is used to calculate the output with "attention". 

#### Conclusion:

During this project, we learnt that a seq2seq model with attention required lots of data and processing power to train. Despite the attention mechanism implemented, the lack of training and the subset of data selected did not allow the model to perform well. Furthermore, there was no hyperparameter tuning done to the model due to the hardware requirements. Our goal with this set up was to get the model to try produce "proper" sentence, which it was able to do to some extent. Also, it is nearly impossible to obtain a human-human like conversation with a chatbot due to the use of a movie lines dataset.

#### Future Direction
- Better hardware to use all available data and train for more epochs, or learn to progressively train the model
- Obtain realistic training data, not movie lines
- BERT embeddings weights used instead of generated own embeddings
- Learn and attempt to implement hierarchical neural attention encoder
- Test Transformer architecture chatbot and compare results

#### References:
- [1]: Dataset collect and information about Cornell movie dialog corpus dataset; Available: https://www.cs.cornell.edu/ cristian/CornellMovieDialogsCorpus.htm
- [2]: Seq2Seq AI Chatbot with Attention Mechanism; Abonia Sojasingarayar; [Paper] Available: https://arxiv.org/ftp/arxiv/papers/2006/2006.02767.pdf
- [3]: Sequence to Sequence Learning with Neural Networks; Ilya Sutskever, Oriol Vinyals, Quoc V. Le; [Paper] Available: https://arxiv.org/pdf/1409.3215v3.pdf
- [4]: Effective Approaches to Attention-based Neural Machine Translation; Minh-Thang Luong, Hieu Pham, Christopher D. Manning; [Paper] Available: https://arxiv.org/pdf/1508.04025.pdf

GitHub URL (see data here): https://github.com/Bilalmajeed1995/Seq2SeqChatbotWithAttention/