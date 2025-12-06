import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def clean_data(data):
    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)

    if 'Unnamed: 32' in data.columns:
        data.drop('Unnamed: 32', axis=1, inplace=True)

    return data

def split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n = len(X)
    n_test = int(test_size * n)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def standard_scaler(X_train, X_test=None):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    std[std == 0] = 1e-6
    
    X_train_scaled = (X_train - mean) / std
    
    if X_test is not None:
        X_test_scaled = (X_test - mean) / std
        return X_train_scaled, X_test_scaled, mean, std
    return X_train_scaled, mean, std

def knn_predict_point(X_train, y_train, x_test_point, k):
    distances = np.sqrt(np.sum((X_train - x_test_point)**2, axis=1))
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = y_train[k_indices]
    counts = np.bincount(k_nearest_labels)
    return np.argmax(counts)

def knn_predict(X_train, y_train, X_test, k):
    y_pred = []
    for x_test_point in X_test:
        y_pred.append(knn_predict_point(X_train, y_train, x_test_point, k))
    return np.array(y_pred)

def cross_validation_accuracy(X, y, k_value, n_folds=5):
    accuracies = []
    fold_size = len(X) // n_folds
    
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        test_indices = np.arange(test_start, test_end)
        train_indices = np.concatenate([np.arange(0, test_start), np.arange(test_end, len(X))])
        
        X_train_cv, y_train_cv = X[train_indices], y[train_indices]
        X_test_cv, y_test_cv = X[test_indices], y[test_indices]
        
        y_pred_cv = knn_predict(X_train_cv, y_train_cv, X_test_cv, k_value)
        
        accuracies.append(get_accuracy(y_pred_cv, y_test_cv))
        
    return np.mean(accuracies)

def get_accuracy(predictions, test_targets):    
    return np.mean(predictions == test_targets)

