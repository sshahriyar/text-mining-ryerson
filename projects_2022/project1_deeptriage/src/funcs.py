#LIBRARIES
import pandas as pd
import numpy as np
import pyarrow as pa

import pickle
import nltk
from collections import Counter
from datasets import dataset_dict
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt

import torch; torch.cuda.is_available()
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

"""GENERAL FUNCTIONS ================================================================================="""
#Generate doc vectors using word2vec and average pooling word vectors
def create_doc_vecs(docs, word_vecs): 
    try:
        word_vec_dim = word_vecs['test'].shape
    except:
        word_vec_dim = word_vecs[0].shape

    doc_vecs = []
    for doc in docs: 
        n_words = 0
        doc_vec = np.zeros(word_vec_dim)
        for word in doc:
            try:
                doc_vec = np.add(doc_vec, word_vecs[word])
                n_words += 1
            except: #Word not found in word2vec vocab
                continue
        doc_vec = doc_vec / n_words
        doc_vecs.append(doc_vec)
        
    return doc_vecs

#TOKENIZATION
def tokenize_data(samples): 
    """Removes punctuation + casefolds(lower)"""
    tokens = []
    for sample in samples: 
        sample = str(sample)
        token_list = nltk.tokenize.RegexpTokenizer("['\w]+").tokenize(sample)
        token_list = [token.lower() for token in token_list]
        tokens.append(token_list)
    return tokens

def filter_data_threshold(full_X, full_y, label_map, n_thresh = 0):
    """TRAINING SAMPLES THRESHOLD: Min. train samples per class (i.e., developer)
    - Determines count of instances per developer and filters accordingly
    - Also resets class indexing for 'y' labels since any n_thresh > 0 will create gaps in label indices
    """
    counts = Counter(full_y)
    labels = [dev for dev,n_samples in dict(counts).items() if n_samples >= n_thresh]
    inds =  [True if dev in labels else False for dev in full_y]
    n_classes = len(labels)

    X_t = full_X[inds]
    y_t = full_y[inds]; y_t = pd.Series(y_t)
    label_map = pd.DataFrame(label_map.items(), columns=['code','email'])
    
    #Map old class indices (with gaps) to new indices (no gaps)
    if n_thresh > 0:
        y_map = {old: new for old, new in zip(list(set(y_t.values)), np.arange(n_classes))}
        y_t.replace(y_map, inplace=True)
        
        #adjust label map to correspond to new class indices
        new_label_map = label_map.query(f'code in {list(y_map.keys())}')
        new_label_map = new_label_map.replace({'code': y_map}).sort_values(by='code')
        label_map = new_label_map.copy()

    # data = pd.concat([y_t, X_t], axis =1)
    X_t.reset_index(drop=True, inplace=True)
    return X_t, y_t, label_map, n_classes

"""LSTM ======================================================================================="""
def lstm_build_embed_mat(embeddings_index, word_index, max_words, embedding_dim = None):
    #Generates embedding matrix for use in input of LSTM model 
    if embedding_dim == None:  
        embedding_dim = len(embeddings_index[list(embeddings_index.keys())[0]]) #sampling 1st item of word vectors
    embedding_mat = np.zeros((max_words, embedding_dim))

    for word, i in word_index.items():
        if i < max_words: 
            embedding_vec = embeddings_index.get(word)
            if embedding_vec is not None: 
                embedding_mat[i] = embedding_vec
    return embedding_dim, embedding_mat

def lstm_processing(train_x, test_x, max_words, max_len): 
    #Preprocessing step for text input for use in LSTM model
    def lstm_build_sequences(data, max_words, max_len, tokenizer = None): 
        if tokenizer == None: #fit new tokenizer
            tokenizer = Tokenizer(num_words=max_words, lower=False)
            tokenizer.fit_on_texts(data)

        sequences = tokenizer.texts_to_sequences(data)
        sequences = pad_sequences(sequences, maxlen = max_len)
        return sequences, tokenizer
    train_x, tokenizer = lstm_build_sequences(train_x, max_words, max_len)
    test_x, tokenizer = lstm_build_sequences(test_x, max_words, max_len, tokenizer=tokenizer)
    word_index = tokenizer.word_index
    return train_x, test_x, word_index

def lstm_train(model, train_x, train_y, epochs, batch, callbacks):
    history = model.fit(
        train_x, np.array(train_y),
        verbose=0,
        epochs=int(epochs),
        batch_size=int(batch),
        validation_split=0.1, 
        callbacks=[callbacks])
    return model, history
    
"""TRAINING/KFOLD =================================================================="""
def get_folds(n_splits=10, stratified=False): 
    kfold_func = StratifiedKFold if stratified else KFold
    kfold = kfold_func(n_splits=n_splits, shuffle=True, random_state=10)
    return kfold

def reindex_labels(train_y_f, test_y_f): 
    """Reindexing of developer labels performed after filtering threshold and chronological CV indexing.
    This is done since both steps create gaps in the developer labels:  
    - Filtering thresholds remove developers with < threshold samples
    - Chronological CV further removes developers present in test set but not in training set"""
    #Setup new developer codes mapping
    all_devs = np.unique(np.concatenate([train_y_f, test_y_f]))
    new_codes = pd.Series(all_devs).astype('category').cat.codes
    label_map = {old: new for old,new in zip(all_devs, new_codes)}
    #Map old devs to new devs
    train_y_f = pd.Series(train_y_f).map(label_map)
    test_y_f = pd.Series(test_y_f).map(label_map)

    return train_y_f, test_y_f

"""PREDICTIONS, METRICS, PLOTS, STORAGE =================================================================="""
def top_n_acc(y_preds, y_true, n=1): 
    """Receives y_preds as sparse categorical pred vector,
        calculates accuracy score for top-n prediction"""
    def top_n_preds(preds, n): 
        top_preds = []
        for pred_arr in preds: 
            top_n = np.argsort(pred_arr)[-n:]
            top_preds.append(top_n)
        return top_preds

    top_preds = top_n_preds(y_preds, n)
    num_samples = len(top_preds)
    n = len(top_preds[0])
    correct = 0
    for pred_arr, y in zip(top_preds, y_true):
        correct += 1 if y in pred_arr else 0
    
    return correct/num_samples

def calc_top_n_acc(y_preds, y_true, results_dict, k, n=1):
    acc = top_n_acc(y_preds, y_true, n)
    results_dict['average'].append(acc)
    results_dict['cv_'+str(k)] = acc

    return results_dict

def init_results_dicts(dataset_name, sample_thresh, clf_name, n_classes, total_samples):
    metrics = {
        'top_n': {
            'func': calc_top_n_acc,
            'vals': [1, 10]}
    }

    results_dicts = []
    for metric, metric_dict in metrics.items():
        if metric_dict['vals']: 
            for val in metric_dict['vals']:
                results_dicts.append({'dataset': dataset_name,
                                        'metric': metric, 
                                        'metric_n': val, 
                                        'threshold': sample_thresh, 
                                        'classifier': clf_name, 
                                        'n_classes': n_classes, 
                                        'total_samples': total_samples,
                                        'average': [], 
                                        })
        else: 
            results_dicts.append({'dataset': dataset_name,
                                    'metric': metric, 
                                    'metric_n': 0, 
                                    'threshold': sample_thresh, 
                                    'classifier': clf_name, 
                                    'n_classes': n_classes, 
                                    'total_samples': total_samples,
                                    'average': [], 
                                    })
    return results_dicts, metrics

def calc_all_metrics(y_preds, y_true, results_dicts, metrics, k): 
    updated_dicts = []
    idx = 0
    for metric, metric_dict in metrics.items(): 
        if metric_dict['vals']: 
            for val in metric_dict['vals']:
                current_results = results_dicts[idx]
                updated_results = metric_dict['func'](y_preds, y_true, current_results, k, n=val)
                updated_dicts.append(updated_results)
                idx += 1
        else: 
            current_results = results_dicts[idx]
            updated_results = metric_dict['func'](y_preds, y_true, current_results, k, n=0)
            updated_dicts.append(updated_results)
            idx += 1
    return updated_dicts

def tf_plot(history, model_save_name, show = 1, save = 0): 
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(0, len(acc))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(epochs, acc, 'bo', label='Training acc')
    ax1.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Acc')
    ax1.legend(loc=1)

    ax2.plot(epochs, loss, 'rx', label='Training loss')
    ax2.plot(epochs, val_loss, 'r', label='Validation loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax1.set_title('Training/validation Accuracy & Loss')
    ax2.legend(loc=0)
    if save: 
        plt.savefig(f'plots/runs/{model_save_name}.png')
    if show: 
        plt.show()
    plt.clf()

def store_results(results_dict): 
    results_path = 'results/results.csv'
    results_df = pd.read_csv(results_path)
    #Drop any existing rows based on [top_n, threshold, classifier] combinations
    dataset = results_df['dataset'] == results_dict['dataset']
    metric = results_df['metric'] == results_dict['metric']
    top_n = results_df['metric_n'] == results_dict['metric_n']
    thresh = results_df['threshold'] == results_dict['threshold']
    clf = results_df['classifier'] == results_dict['classifier']
    drop_mask = dataset & metric & top_n & thresh & clf
    results_df = results_df.drop(results_df[drop_mask].index)

    results = pd.DataFrame(results_dict, index=[0])
    results_df = pd.concat([results_df, results], ignore_index=True)
    results_df.to_csv(results_path, index=False)

def store_model(model, model_name, tf_model):
    model_path = f'models/{model_name}'
    if tf_model:
        model.save(model_path+'.h5', save_format='tf')
    else:
        with open(model_path+'.pkl', 'wb') as f: 
            pickle.dump(model, f)
        