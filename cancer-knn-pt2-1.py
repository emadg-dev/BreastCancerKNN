
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from methods import * 


def main():
        
    try:
        data = pd.read_csv('cancer.csv')
    except FileNotFoundError:
        print("dataset file 'cancer.csv' not found.")
        exit()

    data = clean_data(data)
    
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    X_np = data.drop('diagnosis', axis=1).values
    Y_np = data['diagnosis'].values

    X_train, X_test, y_train, y_test = split(X_np, Y_np)

    X_train_scaled, X_test_scaled, _, _ = standard_scaler(X_train, X_test)

    X_train_clean = X_train_scaled
    y_train_clean = y_train

    X_train_noisy_10, y_train_noisy_10 = noise_injection(X_train_scaled, y_train, noise_rate=0.10)
    X_train_noisy_20, y_train_noisy_20 = noise_injection(X_train_scaled, y_train, noise_rate=0.20)

    report_class_distribution(y_train_clean, "Clean Data (Baseline)")
    report_class_distribution(y_train_noisy_10, "10% Noisy Data")
    report_class_distribution(y_train_noisy_20, "20% Noisy Data")


    # k_values = range(1, 17)
    # accuracies = []

    # for k in k_values:
    #     acc = cross_validation_accuracy(X_train_noisy_20, y_train_noisy_20, k_value=k, n_folds=5)
    #     accuracies.append(acc)
    #     print(f"K={k}  Accuracy={acc:.4f}")

    # best_k = k_values[np.argmax(accuracies)]
    # print(f"Best K = {best_k}")

    # plt.plot(k_values, accuracies)
    # plt.title("Accuracy - K")
    # plt.xlabel("K")
    # plt.ylabel("Accuracy")
    # plt.grid(True)
    # plt.show()

    # print('data description:\n', data.describe())
    # print('data info:\n' , data.info())

def report_class_distribution(y_data, title):
    counts = np.bincount(y_data)
    total = len(y_data)
    
    if len(counts) == 2:
        percent_0 = counts[0] / total * 100
        percent_1 = counts[1] / total * 100
        print(f"--- {title} ---")
        print(f"Class 0 (Benign): {counts[0]} ({percent_0:.1f}%)")
        print(f"Class 1 (Malignant): {counts[1]} ({percent_1:.1f}%)")
    else:
        print(f"Error in class count for {title}")


if __name__ == "__main__":
    main()