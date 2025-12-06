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

    k_values = range(1, 17)
    accuracies = []

    for k in k_values:
        acc = cross_validation_accuracy(X_np, Y_np, k_value=k, n_folds=5)
        accuracies.append(acc)
        print(f"K={k}  Accuracy={acc:.4f}")

    best_k = k_values[np.argmax(accuracies)]
    print(f"Best K = {best_k}")

    plt.plot(k_values, accuracies)
    plt.title("Accuracy - K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    print('data description:\n', data.describe())
    print('data info:\n' , data.info())

    for i in range(1, 17):
        knn_results = knn_predict(X_train, y_train, X_test, i)
        # print('KNN predictions (k=13):')
        # print (knn_results)

        # print('real test targets:')
        # print (y_test)

        print(f'accuracy (k={i}): ', get_accuracy(knn_results, y_test))



if __name__ == "__main__":
    main()