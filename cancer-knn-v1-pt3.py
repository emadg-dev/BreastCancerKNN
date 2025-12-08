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

    mean_features, se_features, worst_features = get_grouped_features(X_df)

    X_mean = get_X_by_features(X_df, mean_features)
    X_se = get_X_by_features(X_df, se_features)
    X_worst = get_X_by_features(X_df, worst_features)
    
    print(f'mean features: {mean_features}')
    print(f'se features: {se_features}')
    print(f'worst features: {worst_features}')

    corr_mean = X_mean.corr()
    corr_se = X_se.corr()
    corr_worst = X_worst.corr()

    plt.figure(figsize=(15, 6))

    # 1. گروه Mean
    plt.subplot(1, 3, 1)
    cax1 = plt.imshow(corr_mean, cmap='coolwarm', interpolation='nearest')
    plt.title('Correlation Matrix (Mean Features)')
    plt.xticks(range(len(corr_mean.columns)), corr_mean.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr_mean.columns)), corr_mean.columns, fontsize=8)
    plt.colorbar(cax1)

    # 2. گروه SE
    plt.subplot(1, 3, 2)
    cax2 = plt.imshow(corr_se, cmap='coolwarm', interpolation='nearest')
    plt.title('Correlation Matrix (SE Features)')
    plt.xticks(range(len(corr_se.columns)), corr_se.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr_se.columns)), corr_se.columns, fontsize=8)
    plt.colorbar(cax2)

    # 3. گروه Worst
    plt.subplot(1, 3, 3)
    cax3 = plt.imshow(corr_worst, cmap='coolwarm', interpolation='nearest')
    plt.title('Correlation Matrix (Worst Features)')
    plt.xticks(range(len(corr_worst.columns)), corr_worst.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr_worst.columns)), corr_worst.columns, fontsize=8)
    plt.colorbar(cax3)

    plt.tight_layout()
    plt.show()




    # X_np = X_df.values
    # Y_np = y_df.values

    # X_train, X_test, y_train, y_test = split(X_np, Y_np)

    # X_train_scaled, X_test_scaled, _, _ = standard_scaler(X_train, X_test)

    # best_k = 6

    # y_pred_proper , accuracy_proper = get_pred_and_accuracy(X_train_scaled, y_train, X_test_scaled, y_test, best_k)
    # print(f'accuracy on properly scaled data: {accuracy_proper:.4f}')

    # X_full = np.concatenate([X_train, X_test])

    # X_full_scaled, _, _ = standard_scaler(X_full)

    # train_length = len(X_train)
    # X_train_leaked = X_full_scaled[:train_length]
    # X_test_leaked = X_full_scaled[train_length:]

    # y_pred_leaked, accuracy_leaked = get_pred_and_accuracy(X_train_leaked, y_train, X_test_leaked, y_test, best_k)

    # print(f'accuracy on leaky scaled data: {accuracy_leaked:.4f}')

    # difference = accuracy_leaked - accuracy_proper
    # print(f'difference on accuracy between proper and leaked scaling: {difference:.4f}')

    # k_values = range(1, 17)
    # accuracies = []
    
    # for k in k_values:
    #     acc = cross_validation_accuracy(X_train_scaled, y_train, k_value=k, n_folds=5)
    #     accuracies.append(acc)

    # best_k = k_values[np.argmax(accuracies)]
    # print(f"Best K = {best_k}")

    # plt.plot(k_values, accuracies)
    # plt.title("Accuracy - K")
    # plt.xlabel("K")
    # plt.ylabel("Accuracy")
    # plt.grid(True)
    # plt.show()

def get_grouped_features(X_df):

    feature_names = X_df.columns.tolist()
    mean_features = [col for col in feature_names if 'mean' in col]
    se_features = [col for col in feature_names if 'se' in col]
    worst_features = [col for col in feature_names if 'worst' in col]

    return mean_features, se_features, worst_features

def get_X_by_features(X_df, feature_group):
    X_feature_group = X_df[feature_group]
    return X_feature_group

def get_pred_and_accuracy(X_train, y_train, X_test, y_test, k):
    y_pred = knn_predict(X_train, y_train, X_test, k)
    accuracy = get_accuracy(y_pred, y_test)
    return y_pred, accuracy

if __name__ == "__main__":
    main()