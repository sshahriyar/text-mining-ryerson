from msilib import sequence
import os
import src.funcs as funcs
import src.processing as p
from src.attention import *
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, BatchNormalization, concatenate, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Wrapper, InputSpec, TimeDistributed
from tensorflow.keras.regularizers import L1, L2
from keras.utils.vis_utils import plot_model

import imp; imp.reload(funcs); imp.reload(p)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) 

"""SETUP ======================================================================="""
MODEL_NAME = 'dbrnna'

MAX_LEN = 200
MAX_WORDS = 10000
MIN_TRAIN_THRESHOLDS = [0, 5, 10, 20]
NUM_K_FOLDS = 10
EPOCHS = 20
BATCH_SIZE = 64

LOSS = 'sparse_categorical_crossentropy'
METRIC = ['acc']
OPTIMIZER = Adam(learning_rate=1e-4)
CALLBACKS = EarlyStopping(monitor='val_loss',
                        patience=3, 
                        restore_best_weights=True)

"""ATTENTION ======================================================================="""
"""ARCHITECTURE"""
def bilstm_model(embedding_dim, embedding_mat, n_classes, lstm_units, layer_units): 

    input = Input(shape=(MAX_LEN,), dtype='float16')
    sequence_embed = Embedding(MAX_WORDS, 
                            embedding_dim, 
                            input_length=MAX_LEN, 
                            weights=[embedding_mat],
                            trainable=False)(input)

    forwards_1 = LSTM(lstm_units, dropout=0.2, return_sequences=True)(sequence_embed)
    attention_1 = SoftAttentionConcat()(forwards_1)
    after_dp_forward_5 = BatchNormalization()(attention_1)

    backwards_1 = LSTM(lstm_units, dropout=0.2, return_sequences=True, go_backwards=True)(sequence_embed)
    attention_2 = SoftAttentionConcat()(backwards_1)
    after_dp_backward_5 = BatchNormalization()(attention_2)
    merged = concatenate([after_dp_forward_5, after_dp_backward_5])
                
    after_merge = Dense(layer_units, activation='relu')(merged)
    after_dp = Dropout(0.4)(after_merge)
    output = Dense(n_classes, activation='softmax')(after_dp)      

    bilstm = Model(inputs=input, outputs=output)
    return bilstm

def run(lstm_units, layer_units):
    #WORD2VEC EMBEDDINGS
    with open(os.path.join(os.getcwd(), 'embeddings/emb_indices/custom_emb_200.pkl'), 'rb') as f: 
        embeddings_index = pickle.load(f)
 
    full_data, _, label_map = p.read_process_data(overwrite=False)
    full_X, full_y = full_data['X'], full_data['y']

    """DATA FILTERING/PROCESSING ======================================================="""
    MODEL_NAME_ = f"{MODEL_NAME}_lstm{lstm_units}_dense{layer_units}"
    for thresh in MIN_TRAIN_THRESHOLDS: 
        X_t, y_t, label_map_f, n_classes = funcs.filter_data_threshold(full_X, full_y, label_map, n_thresh=thresh)
        X_t = np.array(X_t)
        y_t.reset_index(drop=True, inplace=True)

        #Init results dictionaries for storage
        results_dicts, metrics = funcs.init_results_dicts('chromium', thresh, MODEL_NAME_, n_classes, len(X_t))
        
        """Cross-Validation"""
        fold = 1
        kfold = funcs.get_folds(NUM_K_FOLDS)
        for train_index, test_index in kfold.split(X_t):
            print(f"RUNNING: THRESH = {thresh}, KFOLD = {fold} ################################")
            train_x_f, test_x_f = X_t[train_index], X_t[test_index]
            train_y_f, test_y_f = y_t[train_index], y_t[test_index]
            """Data processing + Embeddings"""
            train_x_f, test_x_f = funcs.tokenize_data(train_x_f), funcs.tokenize_data(test_x_f)
            train_x_f, test_x_f, word_index = funcs.lstm_processing(train_x_f, test_x_f, MAX_WORDS, 200)
            embedding_dim, embedding_mat = funcs.lstm_build_embed_mat(embeddings_index, word_index, MAX_WORDS)
            """Training + Metrics"""
            with tf.device('/gpu:0'):
                model = bilstm_model(embedding_dim, embedding_mat, n_classes, lstm_units, layer_units)
                model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRIC)
                model, history = funcs.lstm_train(model, train_x_f, train_y_f, EPOCHS, BATCH_SIZE, CALLBACKS)

            y_preds = model.predict(test_x_f)
            results_dicts = funcs.calc_all_metrics(y_preds, test_y_f, results_dicts, metrics, fold)
            """Store results"""
            model_save_name = f'{MODEL_NAME_}_thresh-{thresh}'
            funcs.tf_plot(history, model_save_name, show=0, save=1)
            if fold == 1: #only save first cv model
                if thresh == 0:
                    plot_model(model, to_file='plots/model_plots/'+MODEL_NAME_+'_model_plot.png', show_shapes=True, show_layer_names=True)
                if thresh == 20:
                    funcs.store_model(model, model_save_name, tf_model=1)
            del model
            fold += 1

        """Store final results"""
        for results in results_dicts: 
            results['average'] = np.mean(results['average'])
            funcs.store_results(results)

if __name__ == "__main__": 
    run(300, 300)
