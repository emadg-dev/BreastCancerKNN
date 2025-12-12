# first part of the problem:
# using non scaled raw data for knn model making

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from methods import * 


def main():
        
    try:
        data = pd.read_csv('cancer.csv')
    except FileNotFoundError:
        print("dataset file 'cnacer.csv' not found.")
        exit()

    data = clean_data(data)
    
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X_df = data.drop('diagnosis', axis=1)
    y_df = data['diagnosis']

    X_df_train, X_df_test, y_df_train, y_df_test = split_df(X_df, y_df)

    corr_mean, corr_se, corr_worst = get_correlation_matrix(X_df_train)
    # show_correlation_matrix(corr_mean, corr_se, corr_worst)

    high_corr_mean = get_highly_correlated_groups(corr_mean)
    high_corr_se = get_highly_correlated_groups(corr_se)
    high_corr_worst = get_highly_correlated_groups(corr_worst)

    print('highly-correlated groups in se features:', high_corr_se)
    print('highly-correlated groups in mean features:', high_corr_mean)
    print('highly-correlated groups in worst features:', high_corr_worst)

    selected_mean = select_best_features(X_df_train, high_corr_mean)
    selected_se = select_best_features(X_df_train, high_corr_se)
    selected_worst = select_best_features(X_df_train, high_corr_worst)

    print("Selected from mean:", selected_mean)
    print("Selected from se:", selected_se)
    print("Selected from worst:", selected_worst)
    drop_mean = get_features_to_drop(high_corr_mean, selected_mean)
    drop_se = get_features_to_drop(high_corr_se, selected_se)
    drop_worst = get_features_to_drop(high_corr_worst, selected_worst)

    features_to_drop = drop_mean + drop_se + drop_worst

    print("Final features to drop:", features_to_drop)
    X_df_cleaned = X_df_train.drop(columns=features_to_drop)

    print("Final features after cleaning:", X_df_cleaned.columns)

    k_values = range(1, 17)
    accuracies = []
    
    for k in k_values:
        acc = cross_validation_accuracy(X_df_cleaned.values, y_df_train.values , k_value=k, n_folds=5)
        accuracies.append(acc)

    best_k = k_values[np.argmax(accuracies)]
    print(f"Best K = {best_k}")

    # plt.plot(k_values, accuracies)
    # plt.title("Accuracy - K")
    # plt.xlabel("K")
    # plt.ylabel("Accuracy")
    # plt.grid(True)
    # plt.show()


def get_features_to_drop(high_corr_groups, selected_features):
    drop_list = []

    for group, selected in zip(high_corr_groups, selected_features):
        for feature in group:
            if feature != selected:
                drop_list.append(feature)

    return drop_list



def select_best_features(X_df, groups):
    selected = []
    for group in groups:
        vars = X_df[group].var()
        best_feature = vars.idxmax()
        selected.append(best_feature)

    return selected

if __name__ == "__main__":
    main()