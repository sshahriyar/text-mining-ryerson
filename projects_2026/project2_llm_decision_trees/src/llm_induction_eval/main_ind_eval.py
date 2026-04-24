import re
import os
import tempfile
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.impute import KNNImputer
from autogluon.tabular import TabularPredictor
from tabpfn import TabPFNClassifier
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

_env_path = Path(__file__).parent.parent.parent / '.env'
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith('TABPFN_TOKEN'):
                _token = _line.split('=', 1)[1].strip().strip('"').strip("'")
                os.environ['TABPFN_TOKEN'] = _token
                break
MODEL_FOLDERS = {
    'gpt-oss': 'gpt_oss',
    'gemma3': 'gemma3',
    'mistral': 'mistral_small3',
    'qwen3': 'qwen3'
}


def inpute_k_neighbours(X_train, X_test, n_neighbors=10):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test


def load_tree_function(filepath):
    with open(filepath, 'r') as f:
        code = f.read()
    code = re.sub(r'```python\s*\n?', '', code)
    code = re.sub(r'```\s*\n?', '', code)
    code = code.strip()
    match = re.search(r'^def\s+(\w+)\s*\(', code, re.MULTILINE)
    func_name = match.group(1)
    local_ns = {}
    exec(code, local_ns)
    return local_ns[func_name]


def evaluate_ind_dataset(dataset: str):
    valid_datasets = ('bankruptcy', 'boxing1', 'boxing2', 'colic', 'creditscore')
    if dataset not in valid_datasets:
        raise ValueError(f"Dataset must be one of {valid_datasets}")
    X = pd.read_csv(f"./data/data_sets/{dataset}/X.csv")
    y = pd.read_csv(f"./data/data_sets/{dataset}/y.csv").squeeze()

    results = []

    for i_split in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=i_split)
        X_train, X_test = inpute_k_neighbours(X_train, X_test, n_neighbors=10)
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        # Get LLM Induction Predictions
        for model_name, model_folder in MODEL_FOLDERS.items():
            functions = [load_tree_function(f"./data/llm_induction/{model_folder}/{dataset}/dt_func_{i}.txt") for i in range(5)]
            preds = []
            for _, row in X_test.iterrows():
                votes = []
                for func in functions:
                    pred, _ = func(row)
                    votes.append(int(pred))
                preds.append(int(sum(votes) > len(votes) / 2))
            score = f1_score(y_test.values, preds, average='macro')
            results.append({
                'model': model_name,
                'f1-score': score
            })

        # AutoGluon
        train_df = X_train.copy().reset_index(drop=True)
        train_df['label'] = y_train.reset_index(drop=True).values
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = TabularPredictor(label='label', path=tmpdir, verbosity=0).fit(
                train_df, time_limit=30
            )
            ag_preds = predictor.predict(X_test.reset_index(drop=True)).values
        score = f1_score(y_test.values, ag_preds, average='macro')
        results.append({'model': 'autogluon', 'f1-score': score})

        # TabPFN
        tabpfn = TabPFNClassifier()
        tabpfn.fit(X_train, y_train)
        score = f1_score(y_test.values, tabpfn.predict(X_test), average='macro')
        results.append({'model': 'tabpfn', 'f1-score': score})

    return pd.DataFrame(results)
