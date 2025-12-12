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

    X_np = data.drop('diagnosis', axis=1).values
    Y_np = data['diagnosis'].values

    X_train, X_test, y_train, y_test = split(X_np, Y_np)

    X_train_scaled, X_test_scaled, _, _ = standard_scaler(X_train, X_test)

    best_k = 6

    y_pred_proper , accuracy_proper = get_pred_and_accuracy(X_train_scaled, y_train, X_test_scaled, y_test, best_k)
    print(f'accuracy on properly scaled data: {accuracy_proper:.4f}')

    X_full = np.concatenate([X_train, X_test])

    X_full_scaled, _, _ = standard_scaler(X_full)

    train_length = len(X_train)
    X_train_leaked = X_full_scaled[:train_length]
    X_test_leaked = X_full_scaled[train_length:]

    y_pred_leaked, accuracy_leaked = get_pred_and_accuracy(X_train_leaked, y_train, X_test_leaked, y_test, best_k)

    print(f'accuracy on leaky scaled data: {accuracy_leaked:.4f}')

    difference = accuracy_leaked - accuracy_proper
    print(f'difference on accuracy between proper and leaked scaling: {difference:.4f}')

    k_values = range(1, 17)
    accuracies = []
    
    for k in k_values:
        acc = cross_validation_accuracy(X_train_scaled, y_train, k_value=k, n_folds=5)
        accuracies.append(acc)

    best_k = k_values[np.argmax(accuracies)]
    print(f"Best K = {best_k}")

    # plt.plot(k_values, accuracies)
    # plt.title("Accuracy - K")
    # plt.xlabel("K")
    # plt.ylabel("Accuracy")
    # plt.grid(True)
    # plt.show()



def get_pred_and_accuracy(X_train, y_train, X_test, y_test, k):
    y_pred = knn_predict(X_train, y_train, X_test, k)
    accuracy = get_accuracy(y_pred, y_test)
    return y_pred, accuracy

if __name__ == "__main__":
    main()