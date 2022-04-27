#LIBRARIES ==========================================================================
import os
import pandas as pd
import numpy as np
import string
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
import tqdm
from gensim.models import Word2Vec

#CONTROL ==========================================================================
"""D1 = untriaged bugs, D2 = triaged bugs"""
RESULTS_PATH = 'results/'
EMB_DIR = 'embeddings/'

#FUNCTIONS ==========================================================================
def create_doc_vecs(emb_ind, docs, dim=100): 
    #Generates average document vectors for word2vec model
    doc_vecs = []
    for doc in docs: 
        n_words = 0
        doc_vec = np.zeros(dim)
        for word in doc:
            try:
                doc_vec = np.add(doc_vec, emb_ind[word])
                n_words += 1
            except: #Word not found in word2vec vocab
                continue
        doc_vec = doc_vec / n_words
        doc_vecs.append(doc_vec)
    return doc_vecs

def text_processing(X): 
    """Processing implementation from Deeptriage paper: **
    - remove urls, stack traces, hex code, lowercase, tokenize, strip
    """
    all_data = []
    for idx, item in X.iterrows():
        #1. Remove \r 
        current_title = item['issue_title'].replace('\r', ' ')
        current_desc = item['description'].replace('\r', ' ')    
        #2. Remove URLs
        current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', current_desc)    
        #3. Remove Stack Trace
        start_loc = current_desc.find("Stack trace:")
        current_desc = current_desc[:start_loc]    
        #4. Remove hex code
        current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
        current_title= re.sub(r'(\w+)0x\w+', '', current_title)    
        #5. Change to lower case
        current_desc = current_desc.lower()
        current_title = current_title.lower()    
        #6. Tokenize
        current_desc_tokens = nltk.word_tokenize(current_desc)
        current_title_tokens = nltk.word_tokenize(current_title)
        #7. Strip trailing punctuation marks    
        current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
        current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]      
        #8. Join the lists
        current_data = current_title_filter + current_desc_filter
        current_data = ' '.join(current_data)
        all_data.append(current_data)  
    return all_data

def read_process_data(overwrite, triaged_frac=0.2, untriaged_frac=0.3):
    """Data features
    - owner: developer assigned to bug (empty if unresolved bug)
    - issue_title: summary/title of the bug report
    - description: detailed description of bug with urls, code, text, etc. 

    @overwrite: if True, read raw data and perform processing, otherwise, read processed data
    """
    triaged_path = os.path.join(os.getcwd(), 'data\\triaged.csv')
    untriaged_path = os.path.join(os.getcwd(), 'data\\untriaged.csv')
    label_map_path = os.path.join(os.getcwd(), 'data\\label_map.csv')

    if not overwrite:
        triaged = pd.read_csv(triaged_path, index_col=False)
        untriaged = pd.read_csv(untriaged_path, index_col=False)
        label_map = pd.read_csv(label_map_path, index_col=False)
    else:
        #TRIAGED BUGS ===================================================
        file_path_triaged = os.path.join(os.getcwd(), 'data\\classifier_data_0.csv')
        df = pd.read_csv(file_path_triaged); df = df.sample(frac = triaged_frac)
        df = df.applymap(str)
        ##Labels
        y = df['owner'].astype('category').cat.codes; y.name='label'
        y.reset_index(drop=True, inplace=True)
        label_map = {code: label for label, code in zip(df['owner'], y)}
        label_map = pd.DataFrame.from_dict({
                                    'code': label_map.keys(), 
                                    'label': label_map.values()})\
                                        .sort_values(by='code')\
                                        .reset_index(drop=True)
        ##Text: processing
        X = df.drop('owner', axis=1)
        X = text_processing(X)
        triaged = pd.DataFrame({'X': X, 'y': y.values})

        #UNTRIAGED BUGS ===================================================
        file_path_untriaged = os.path.join(os.getcwd(), 'data\\deep_data.csv')
        df_ = pd.read_csv(file_path_untriaged, usecols=['issue_title','description'])
        df_ = df_.sample(frac = untriaged_frac)
        df_ = df_.applymap(str)
        ##Text: processing
        untriaged = text_processing(df_)
        untriaged = pd.Series(untriaged)

        #SAVE
        triaged.to_csv(triaged_path, index=False)
        untriaged.to_csv(untriaged_path, index=False)
        label_map.to_csv(label_map_path, index=False)

    return triaged, untriaged, label_map

#EMBEDDINGS
def build_custom_emb_model(data, vector_size=200, min_count=5, window=5, overwrite=False):
    if overwrite: 
        data = [word_tokenize(x) for x in data]
        vec_model = Word2Vec(sentences = data, 
                            min_count = min_count, 
                            vector_size = vector_size, 
                            window=window)
        vec_model.save(f'embeddings/word2vec_{vector_size}.model')

def build_emb_indices(): 
    #Load the saved word2vec model to generate word vector index
    word_vecs = Word2Vec.load(f'embeddings/word2vec_200.model')
    embeddings_index = {}
    for word in word_vecs.wv.key_to_index:
        embeddings_index[word] = word_vecs.wv[word]
    #Save embeddings_index for bilstm usage
    with open(os.path.join(os.getcwd(), 'embeddings/emb_indices/custom_emb_200.pkl'), 'wb') as f: 
        pickle.dump(embeddings_index, f)

## Results summary df: to hold all results of all model runs
def build_results_dfs(overwrite): 
    if overwrite: 
        path = os.path.join(os.getcwd(), 'results/')
        if not os.path.exists(path): 
            os.mkdir(path)
        cols = ['dataset', 'metric', 'metric_n', 'threshold', 'classifier', 'n_classes', 'total_samples']
        cols = cols+['cv_'+str(+n) for n in range(1, 10+1)]+['average']
        results = pd.DataFrame(columns = cols)
        results.to_csv(path + '/results.csv', index=False)
