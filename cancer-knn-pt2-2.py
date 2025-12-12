
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

    # Find optimal k values
    optimal_clean_k = get_optimal_k(X_train_clean, y_train_clean)
    optimal_noisy_10_k = get_optimal_k(X_train_noisy_10, y_train_noisy_10)
    optimal_noisy_20_k = get_optimal_k(X_train_noisy_20, y_train_noisy_20)

    # Evaluate on k=1, optimal_k, 16 for all datasets
    results_clean = evaluate_models_on_test(
        X_train_clean, y_train_clean, X_test_scaled, y_test, [1, optimal_clean_k, 16]
    )
    results_noisy_10 = evaluate_models_on_test(
        X_train_noisy_10, y_train_noisy_10, X_test_scaled, y_test, [1, optimal_noisy_10_k, 16]
    )
    results_noisy_20 = evaluate_models_on_test(
        X_train_noisy_20, y_train_noisy_20, X_test_scaled, y_test, [1, optimal_noisy_20_k, 16]
    )

    # Print all results at the end
    print("\nclean_dataset:")
    for k in [1, optimal_clean_k, 16]:
        k_title = k
        if k == optimal_clean_k: 
            k_title=f"k={k} (opt)"
        print(f"k={k_title} -> accuracy={results_clean[k]:.4f}")

    print("\nnoisy_10_dataset:")
    for k in [1, optimal_noisy_10_k, 16]:
        k_title = k
        if k == optimal_noisy_10_k:
            k_title=f"k={k} (opt)"
        print(f"k={k_title} -> accuracy={results_noisy_10[k]:.4f}")

    print("\nnoisy_20_dataset:")
    for k in [1, optimal_noisy_20_k, 16]:
        k_title = k
        if k == optimal_noisy_20_k:
            k_title=f"k={k} (opt)"
        print(f"k={k_title} -> accuracy={results_noisy_20[k]:.4f}")


    plot_noise_impact(results_clean, optimal_clean_k, results_noisy_10, optimal_noisy_10_k, results_noisy_20, optimal_noisy_20_k)
    # plt.plot(k_values, accuracies)
    # plt.title("Accuracy - K")
    # plt.xlabel("K")
    # plt.ylabel("Accuracy")
    # plt.grid(True)
    # plt.show()

    # print('data description:\n', data.describe())
    # print('data info:\n' , data.info())




def evaluate_models_on_test(X_train_data, y_train_data, X_test, y_test, k_values):
    results = {}
    
    for k in k_values:
        y_pred = knn_predict(X_train_data, y_train_data, X_test, k)
        accuracy = get_accuracy(y_pred, y_test)
        results[k] = accuracy
    return results

if __name__ == "__main__":
    main()