from sklearn.ensemble import RandomTreesEmbedding
import pandas as pd
from .config import Config
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline

config = Config()

def rt_embeddings(X_train, y_train, X_test):
    model = RandomTreesEmbedding(
        n_estimators=config.num_trees,
        max_depth=config.max_tree_depth,
        random_state=config.seed
    ).fit(X_train)
    
    emb_transformer = FunctionTransformer(
        lambda X: model.transform(X)
    )
    
    emb_train = pd.DataFrame(emb_transformer.transform(X_train).toarray())
    emb_test = pd.DataFrame(emb_transformer.transform(X_test).toarray())

    return emb_train, emb_test

def et_embeddings(X_train, y_train, X_test, self_supervised=True):
    if self_supervised:
        model = ExtraTreesRegressor(
            max_depth=config.max_tree_depth,
            n_estimators=config.num_trees,
            random_state=config.seed).fit(X_train, X_train)
    else:
        model = ExtraTreesClassifier(
            max_depth=config.max_tree_depth,
            n_estimators=config.num_trees,
            random_state=config.seed
        ).fit(X_train, y_train)
        
    embs = FunctionTransformer(lambda x: model.apply(x))
    scaler = MinMaxScaler()
    transforms = Pipeline([('embedding', embs), ('scaler', scaler)])
    
    train_emb = pd.DataFrame(transforms.fit_transform(X_train))
    test_emb = pd.DataFrame(transforms.transform(X_test))
    return train_emb, test_emb

def rf_embeddings(X_train, y_train, X_test, self_supervised=True):
    if self_supervised:
        model = RandomForestRegressor(
            max_depth=config.max_tree_depth,
            n_estimators=config.num_trees,
            random_state=config.seed
        ).fit(X_train, X_train)
    else:
        model = RandomForestClassifier(
             max_depth=config.max_tree_depth,
            n_estimators=config.num_trees,
            random_state=config.seed
        ).fit(X_train, y_train)
    
    embs = FunctionTransformer(lambda x: model.apply(x))
    scaler = MinMaxScaler()
    transforms = Pipeline([('embedding', embs), ('scaler', scaler)])
    
    train_emb = pd.DataFrame(transforms.fit_transform(X_train))
    test_emb = pd.DataFrame(transforms.transform(X_test))
    return train_emb, test_emb

def xgb_embeddings(X_train, y_train, X_test, self_supervised=True):
    if self_supervised:
        model = XGBRegressor(
            n_estimators=config.num_trees,
            max_depth=config.num_trees,
            random_state=config.seed
        ).fit(X_train, X_train)
    else:
        model = XGBClassifier(
            n_estimators=config.num_trees,
            max_depth=config.num_trees,
            random_state=config.seed
        ).fit(X_train, y_train)
        
    embs = FunctionTransformer(lambda x: model.apply(x))
    scaler = MinMaxScaler()
    transforms = Pipeline([('embedding', embs), ('scaler', scaler)])
    
    train_emb = pd.DataFrame(transforms.fit_transform(X_train))
    test_emb = pd.DataFrame(transforms.transform(X_test))
    
    return train_emb, test_emb

    
