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

    if 'id' in data.columns:
        data.drop('id', axis=1, inplace=True)

    if 'Unnamed: 32' in data.columns:
        data.drop('Unnamed: 32', axis=1, inplace=True)

    # if data['diagnosis'].dtype == 'O':
    #     data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})


    X_df = data.drop('diagnosis', axis=1)
    Y_df = data['diagnosis']

    X_np = X_df.values
    Y_np = Y_df.values

    X_train, X_test, y_train, y_test = split(X_np, Y_np)

    k_values = range(1, 17)
    accuracies = []

    for k in k_values:
        acc = cv_accuracy(X_np, Y_np, k_value=k, n_folds=5)
        accuracies.append(acc)
        print(f"K={k}  Accuracy={acc:.4f}")

    # پیدا کردن بهترین K
    best_k = k_values[np.argmax(accuracies)]
    print(f"Best K = {best_k}")

    # رسم نمودار Accuracy بر حسب K
    plt.plot(k_values, accuracies, marker='o')
    plt.title("Accuracy vs K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    print('data description:\n', data.describe())
    print('data info:\n' , data.info())

    print (knn_predict(X_train, y_train, X_test, 13))
    print()

if __name__ == "__main__":
    main()