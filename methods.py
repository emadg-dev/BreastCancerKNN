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

def split_df(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n = len(X)
    n_test = int(test_size * n)
    indices = np.arange(n)
    np.random.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True)
    )


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

def get_correlation_matrix(X_df):
    mean_features, se_features, worst_features = get_grouped_features(X_df)

    X_mean = get_X_by_features(X_df, mean_features)
    X_se = get_X_by_features(X_df, se_features)
    X_worst = get_X_by_features(X_df, worst_features)
    
    # print(f'mean features: {mean_features}')
    # print(f'se features: {se_features}')
    # print(f'worst features: {worst_features}')

    corr_mean = X_mean.corr()
    corr_se = X_se.corr()
    corr_worst = X_worst.corr()

    return corr_mean, corr_se, corr_worst

def show_correlation_matrix(corr_mean, corr_se, corr_worst):
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 3, 1)
    cax1 = plt.imshow(corr_mean, cmap='coolwarm', interpolation='nearest')
    plt.title('Correlation Matrix (Mean Features)')
    plt.xticks(range(len(corr_mean.columns)), corr_mean.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr_mean.columns)), corr_mean.columns, fontsize=8)
    plt.colorbar(cax1)

    plt.subplot(1, 3, 2)
    cax2 = plt.imshow(corr_se, cmap='coolwarm', interpolation='nearest')
    plt.title('Correlation Matrix (SE Features)')
    plt.xticks(range(len(corr_se.columns)), corr_se.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr_se.columns)), corr_se.columns, fontsize=8)
    plt.colorbar(cax2)

    plt.subplot(1, 3, 3)
    cax3 = plt.imshow(corr_worst, cmap='coolwarm', interpolation='nearest')
    plt.title('Correlation Matrix (Worst Features)')
    plt.xticks(range(len(corr_worst.columns)), corr_worst.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr_worst.columns)), corr_worst.columns, fontsize=8)
    plt.colorbar(cax3)

    plt.tight_layout()
    plt.show()  


def get_highly_correlated_groups(corr_matrix, threshold=0.85):

    visited = set()
    groups = []

    for col in corr_matrix.columns:
        if col in visited:
            continue
        high_corr = corr_matrix.index[
            (corr_matrix[col].abs() >= threshold) & (corr_matrix.index != col)
        ].tolist()

        if high_corr:
            group = [col] + high_corr

            for f in group:
                visited.add(f)

            groups.append(group)

    return groups



def get_grouped_features(X_df):

    feature_names = X_df.columns.tolist()
    mean_features = [col for col in feature_names if 'mean' in col]
    se_features = [col for col in feature_names if 'se' in col]
    worst_features = [col for col in feature_names if 'worst' in col]

    return mean_features, se_features, worst_features

def get_X_by_features(X_df, feature_group):
    X_feature_group = X_df[feature_group]
    return X_feature_group
