from .gemma3 import (
    Gemma3BankruptcyEmbedding,
    Gemma3Boxing1Embedding,
    Gemma3Boxing2Embedding,
    Gemma3ColicEmbedding,
    Gemma3CreditscoreEmbedding
)

from .gpt_oss import (
    GptCreditscoreEmbedding,
    GptBankruptcyEmbedding,
    GptBoxing1Embedding,
    GptBoxing2Embedding,
    GptColicEmbedding
)

from .mistral import (
    MistralCreditscoreEmbedding,
    MistralBankruptcyEmbedding,
    MistralBoxing1Embedding,
    MistralBoxing2Embedding,
    MistralColicEmbedding
)

from .qwen import (
    QwenColicEmbedding,
    QwenBankruptcyEmbedding,
    QwenBoxing1Embedding,
    QwenBoxing2Embedding,
    QwenCreditscoreEmbedding
)
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from ..preprocessing import inpute_k_neighbours
from .other_trees import rt_embeddings, rf_embeddings, et_embeddings, xgb_embeddings
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
MLP_HIDDEN_STATES = [(10,), (25,), (50,), (75,), (100,)]
MLP_ALPHAS = [0.0001, 0.001, 0.01, 0.1]

bankruptcy_models = {'gemma3' : Gemma3BankruptcyEmbedding(), 
                     'gpt-oss': GptBankruptcyEmbedding(), 
                     'mistral': MistralBankruptcyEmbedding(), 
                     'qwen3': QwenBankruptcyEmbedding()}

boxing1_models = {
    'gemma3': Gemma3Boxing1Embedding(),
    'gpt-oss': GptBoxing1Embedding(),
    'mistral': MistralBoxing1Embedding(),
    'qwen3': QwenBoxing1Embedding()
}

boxing2_models = {
    'gemma3': Gemma3Boxing2Embedding(),
    'gpt-oss': GptBoxing2Embedding(),
    'mistral': MistralBoxing2Embedding(),
    'qwen3': QwenBoxing2Embedding(),
}

colic_models = {
    'gemma3': Gemma3ColicEmbedding(),
    'gpt-oss': GptColicEmbedding(),
    'mistral': MistralColicEmbedding(),
    'qwen3': QwenColicEmbedding()
}

credit_models = {
    'gemma3': Gemma3CreditscoreEmbedding(),
    'gpt-oss': GptCreditscoreEmbedding(),
    'mistral': MistralCreditscoreEmbedding(),
    'qwen3': QwenCreditscoreEmbedding()
}

DATASET_MODELS = {
    'bankruptcy': bankruptcy_models,
    'boxing1': boxing1_models,
    'boxing2': boxing2_models,
    'colic': colic_models,
    'creditscore': credit_models
}


def train_mlp(X_train, y_train, X_test, y_test, random_state):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    mlp = MLPClassifier(random_state=random_state, max_iter=1000)
    
    param_grid = {
        "hidden_layer_sizes": MLP_HIDDEN_STATES,
        "alpha": MLP_ALPHAS
    }
    
    grid = GridSearchCV(
        estimator=mlp,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    score = f1_score(y_test, y_pred, average="macro")
    return score

def evaluate_emb_dataset(dataset: str):
    valid_datasets = ('bankruptcy', 'boxing1', 'boxing2', 'colic', 'creditscore')
    if dataset not in valid_datasets:
        raise ValueError(f"Dataset must be one of {valid_datasets}")
    X = pd.read_csv(f"./data/data_sets/{dataset}/X.csv")
    y = pd.read_csv(f"./data/data_sets/{dataset}/y.csv").squeeze()
    
    models = DATASET_MODELS[dataset]
    results = []
    
    for i_split in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=i_split)
        X_train, X_test = inpute_k_neighbours(X_train, X_test, n_neighbors=10)
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)
        
        # Get LLM Embeddings
        train_embs = get_emb(X_train, models)
        test_embs = get_emb(X_test, models)
        for model_name in models.keys():
            X_train_extended = pd.concat([X_train.reset_index(drop=True), train_embs[model_name].reset_index(drop=True)], axis=1)
            X_test_extended = pd.concat([X_test.reset_index(drop=True), test_embs[model_name].reset_index(drop=True)], axis=1)
            X_train_extended.columns = X_train_extended.columns.astype(str)
            X_test_extended.columns = X_test_extended.columns.astype(str)
            
            score = train_mlp(X_train_extended, y_train, X_test_extended, y_test, random_state=i_split)
            results.append({
                'model': model_name,
                'f1-score': score
            })
        # Get Other tree embeddings
        rt_train, rt_test = rt_embeddings(X_train, y_train, X_test)
        rf_s_train, rf_s_test = rf_embeddings(X_train, y_train, X_test, self_supervised=False)
        rf_ss_train, rf_ss_test = rf_embeddings(X_train, y_train, X_test, self_supervised=True)
        et_s_train, et_s_test = et_embeddings(X_train, y_train, X_test, self_supervised=False)
        et_ss_train, et_ss_test = et_embeddings(X_train, y_train, X_test, self_supervised=True)
        xgb_s_train, xgb_s_test = xgb_embeddings(X_train, y_train, X_test, self_supervised=False)
        xgb_ss_train, xgb_ss_test = xgb_embeddings(X_train, y_train, X_test, self_supervised=True)
        
        # concat and train rt
        X_train_extended = pd.concat([X_train.reset_index(drop=True), rt_train.reset_index(drop=True)], axis=1)
        X_test_extended = pd.concat([X_test.reset_index(drop=True), rt_test.reset_index(drop=True)], axis=1)
        X_train_extended.columns = X_train_extended.columns.astype(str)
        X_test_extended.columns = X_test_extended.columns.astype(str)
        score = train_mlp(X_train_extended, y_train, X_test_extended, y_test, random_state=i_split)
        results.append({'model': 'rt', 'f1-score': score})
        
        # concat and train RandomForest Supervised
        X_train_extended = pd.concat([X_train.reset_index(drop=True), rf_s_train.reset_index(drop=True)], axis=1)
        X_test_extended = pd.concat([X_test.reset_index(drop=True), rf_s_test.reset_index(drop=True)], axis=1)
        X_train_extended.columns = X_train_extended.columns.astype(str)
        X_test_extended.columns = X_test_extended.columns.astype(str)
        score = train_mlp(X_train_extended, y_train, X_test_extended, y_test, random_state=i_split)
        results.append({'model': 'rf-s', 'f1-score': score})
        
        # concat and train RandomForest SelfSupervised
        X_train_extended = pd.concat([X_train.reset_index(drop=True), rf_ss_train.reset_index(drop=True)], axis=1)
        X_test_extended = pd.concat([X_test.reset_index(drop=True), rf_ss_test.reset_index(drop=True)], axis=1)
        X_train_extended.columns = X_train_extended.columns.astype(str)
        X_test_extended.columns = X_test_extended.columns.astype(str)
        score = train_mlp(X_train_extended, y_train, X_test_extended, y_test, random_state=i_split)
        results.append({'model': 'rf-ss', 'f1-score': score})
        
        # concat and train ExtraTrees Supervised
        X_train_extended = pd.concat([X_train.reset_index(drop=True), et_s_train.reset_index(drop=True)], axis=1)
        X_test_extended = pd.concat([X_test.reset_index(drop=True), et_s_test.reset_index(drop=True)], axis=1)
        X_train_extended.columns = X_train_extended.columns.astype(str)
        X_test_extended.columns = X_test_extended.columns.astype(str)
        score = train_mlp(X_train_extended, y_train, X_test_extended, y_test, random_state=i_split)
        results.append({'model': 'et-s', 'f1-score': score})
        
        # concat and train ExtraTrees SelfSupervised
        X_train_extended = pd.concat([X_train.reset_index(drop=True), et_ss_train.reset_index(drop=True)], axis=1)
        X_test_extended = pd.concat([X_test.reset_index(drop=True), et_ss_test.reset_index(drop=True)], axis=1)
        X_train_extended.columns = X_train_extended.columns.astype(str)
        X_test_extended.columns = X_test_extended.columns.astype(str)
        score = train_mlp(X_train_extended, y_train, X_test_extended, y_test, random_state=i_split)
        results.append({'model': 'et-ss', 'f1-score': score})
        
        # concat and train XGB Supervised
        X_train_extended = pd.concat([X_train.reset_index(drop=True), xgb_s_train.reset_index(drop=True)], axis=1)
        X_test_extended = pd.concat([X_test.reset_index(drop=True), xgb_s_test.reset_index(drop=True)], axis=1)
        X_train_extended.columns = X_train_extended.columns.astype(str)
        X_test_extended.columns = X_test_extended.columns.astype(str)
        score = train_mlp(X_train_extended, y_train, X_test_extended, y_test, random_state=i_split)
        results.append({'model': 'xgb-s', 'f1-score': score})
        
        # concat and train XGB SelfSupervised
        X_train_extended = pd.concat([X_train.reset_index(drop=True), xgb_ss_train.reset_index(drop=True)], axis=1)
        X_test_extended = pd.concat([X_test.reset_index(drop=True), xgb_ss_test.reset_index(drop=True)], axis=1)
        X_train_extended.columns = X_train_extended.columns.astype(str)
        X_test_extended.columns = X_test_extended.columns.astype(str)
        score = train_mlp(X_train_extended, y_train, X_test_extended, y_test, random_state=i_split)
        results.append({'model': 'xgb-ss', 'f1-score': score})
        
        # baseline
        score = train_mlp(
            X_train,
            y_train,
            X_test,
            y_test,
            random_state=i_split
        )
        
        results.append({
            'model': 'baseline',
            'f1-score': score
        })
    return pd.DataFrame(results)

def get_emb(X: pd.DataFrame, models: dict):
    total_emb = {key: [] for key in models.keys()}
    
    for _, row in X.iterrows():
        for model_name, model in models.items():
            current_embed = []
            for runner in model.runner:                      
                _, emb = runner(row)
                current_embed.extend(emb)
            total_emb[model_name].append(current_embed)
    emb_dfs = {
        model_name: pd.DataFrame(emb_rows, index=X.index)
        for model_name, emb_rows in total_emb.items()
    }
    return emb_dfs