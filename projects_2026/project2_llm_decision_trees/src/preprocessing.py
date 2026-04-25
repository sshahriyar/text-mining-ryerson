from sklearn.impute import KNNImputer

'''
Preprocessing from the paper that imputs missing values with nearest 10-neighbors method
'''

def inpute_k_neighbours(X_train, X_test, n_neighbors=10):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    return X_train, X_test