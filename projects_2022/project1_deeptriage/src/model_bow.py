import src.funcs as funcs
import src.processing as p

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import imp; imp.reload(funcs); imp.reload(p)

"""SETUP ======================================================================="""
MODEL = 'bow'

MIN_TRAIN_THRESHOLDS = [0, 5, 10, 20]
NUM_K_FOLDS = 10

"""FUNCTIONS"""
def bow(train_data): 
    tfidf = TfidfTransformer(use_idf=False)
    vec = CountVectorizer(min_df=0.1, max_df=0.90, max_features=10000, dtype=np.int16)
    
    train_data = train_data.astype('U').tolist()
    counts = vec.fit_transform(train_data)
    feats = tfidf.fit_transform(counts)
    return feats

def get_ordered_preds(clf, y_preds, n_classes, cos = False):
    """Purpose: to prevent misaligned prediction indices in cases where
    train_y data doesn't contain all classes (occuring when min. train threshold == 0 
    as there are several labels with 1 instance). 
    E.g., when n_classes == 212, train_y may only contain < 212 instances and 
        sklearn's .predict_proba() method would output an array with < 212 columns 
        which is unintended. 
    Solution: for any missing class 'i', fill in array with zeros at column 'i' as placeholder
    """
    n_samples = y_preds.shape[0]
    # class_pred_labels = clf.columns if cos else clf.classes_
    class_pred_labels = clf.columns if cos else clf.classes_
    all_labels = np.arange(n_classes)
    
    ordered_preds = np.zeros(shape=(n_samples, n_classes))
    for col in range(y_preds.shape[1]):
        preds = y_preds[:, col]
        correct_class = class_pred_labels[col]
        #update existing class with prob. preds; otherwise, leave as 0 vals
        if correct_class in all_labels: 
            ordered_preds[:, correct_class] = preds
    return ordered_preds

#MODELS ====================================================================
def MNB(train_x, train_y, test_x, test_y, n_classes): 
    clf = MultinomialNB(alpha=0.01)        
    clf = OneVsRestClassifier(clf).fit(train_x, train_y)
    y_preds=clf.predict_proba(test_x)
    y_preds = get_ordered_preds(clf, y_preds, n_classes)
    acc = accuracy_score(test_y, np.argmax(y_preds, axis=1))

    return y_preds, acc, clf

def SVM(train_x, train_y, test_x, test_y, n_classes): 
    clf = SVC(kernel='rbf', probability=True, verbose=False, decision_function_shape='ovr')
    clf = clf.fit(train_x, train_y)
    y_preds=clf.predict_proba(test_x)
    y_preds = get_ordered_preds(clf, y_preds, n_classes)
    acc = accuracy_score(test_y, np.argmax(y_preds, axis=1))

    return y_preds, acc, clf

def COS(train_x, train_y, test_x, test_y, n_classes): 
    predict = cosine_similarity(test_x, train_x)
    classes = np.array(train_y)
    y_preds = []
    for pred in predict:
        y_preds.append(list(pred/sum(pred)))
    classifierModel = pd.DataFrame(y_preds)
    classifierModel.columns = classes
    y_preds_ordered = get_ordered_preds(classifierModel, np.array(y_preds), n_classes, cos=True)
    y_preds_labels = np.argmax(y_preds_ordered, axis=1)
    accuracy = accuracy_score(test_y, y_preds_labels)

    return y_preds_labels, accuracy, classifierModel

def LOG(train_x, train_y, test_x, test_y, n_classes): 
    scaler = StandardScaler(with_mean=False).fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    clf = LogisticRegression(solver='lbfgs', penalty='l2', tol=0.01, max_iter=10000)
    clf = OneVsRestClassifier(clf).fit(train_x, train_y)
    y_preds=clf.predict_proba(test_x)
    y_preds = get_ordered_preds(clf, y_preds, n_classes)
    acc = accuracy_score(test_y, np.argmax(y_preds, axis=1))

    return y_preds, acc, clf

def run():
    model_funcs = {
        'mnb': {'model_func': MNB, 'run': True}, 
        'cos': {'model_func': COS, 'run': False}, 
        'log': {'model_func': LOG, 'run': True},
        'svm': {'model_func': SVM, 'run': False} 
        }

    full_data, _, label_map = p.read_process_data(overwrite=False)
    full_X, full_y = full_data['X'], full_data['y']
    #MODELING, CV
    for model, model_dict in {k: v for k,v in model_funcs.items() if v['run']}.items(): 
        MODEL_NAME = MODEL+f'_{model}'
        model_func = model_dict['model_func']
        """DATA FILTERING/PROCESSING ======================================================="""
        for thresh in MIN_TRAIN_THRESHOLDS:
            X_t, y_t, label_map_f, n_classes = funcs.filter_data_threshold(full_X, full_y, label_map, n_thresh=thresh)
            X_t = bow(X_t).toarray()
            y_t.reset_index(drop=True, inplace=True)

            #Init results dictionaries for storage
            results_dicts, metrics = funcs.init_results_dicts('chromium', thresh, MODEL_NAME, n_classes, len(X_t))

            """Cross-Validation"""
            fold = 1
            kfold = funcs.get_folds(NUM_K_FOLDS)
            for train_index, test_index in kfold.split(X_t):
                print(f"RUNNING {MODEL_NAME}: THRESH = {thresh}, KFOLD = {fold} ################################")
                train_x_f, test_x_f = X_t[train_index], X_t[test_index]
                train_y_f, test_y_f = y_t[train_index], y_t[test_index]

                """Training + Metrics"""
                y_preds, acc, clf = model_func(train_x_f, train_y_f, test_x_f, test_y_f, n_classes)
                results_dicts = funcs.calc_all_metrics(y_preds, test_y_f, results_dicts, metrics, fold)

                """Store results"""
                model_save_name = f'{MODEL_NAME}_thresh-{thresh}' #save last model after all CVs
                if fold == 1 and thresh == 20: 
                    funcs.store_model(clf, model_save_name, tf_model=0)
                del clf
                fold += 1

            """Store final results"""
            for results in results_dicts: 
                results['average'] = np.mean(results['average'])
                funcs.store_results(results)

if __name__ == "__main__": 
    run()