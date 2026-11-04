"""Run Qiskit VQC experiments separately from classical ML.

Run:
    python diabetes_vqc.py

The VQC can be computationally expensive. CV_SPLITS and VQC_MAXITER are
kept at the initial-test values from the original script.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC

from diabetes_common import (
    RANDOM_STATE, CV_SPLITS, FEATURE_CONFIGS, load_dataset,
    preprocess_data, calculate_metrics, save_results
)

VQC_MAXITER = 5  # Increase for final experiments only after timing tests.


def create_vqc(num_features):
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = RealAmplitudes(num_qubits=num_features, reps=1)
    sampler = StatevectorSampler(default_shots=1024, seed=RANDOM_STATE)
    optimizer = COBYLA(maxiter=VQC_MAXITER)
    return VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler
    )


def main():
    X, y = load_dataset()
    print("Dataset shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())

    cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    results = []

    for config in FEATURE_CONFIGS:
        print(f"\nFeature selection: {config['name']}, k={config['k']}")
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            X_train_sel, X_test_sel = preprocess_data(
                X_train, y_train, X_test,
                config["selector"], config["k"]
            )
            n_features = X_train_sel.shape[1]

            # Qiskit feature encoding expects values in [0, pi].
            scaler = MinMaxScaler(feature_range=(0, np.pi))
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            model = create_vqc(num_features=n_features)

            start = time.time()
            model.fit(X_train_scaled, y_train.to_numpy())
            train_time = time.time() - start

            start = time.time()
            y_pred = np.asarray(model.predict(X_test_scaled)).astype(int).ravel()
            probabilities = np.asarray(model.predict_proba(X_test_scaled))

            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                y_prob = probabilities[:, 1]
            else:
                y_prob = probabilities.ravel()
            predict_time = time.time() - start

            metrics = calculate_metrics(y_test, y_pred, y_prob)
            circuit_depth = (
                model.feature_map.depth() + model.ansatz.depth()
            )
            row = {
                "Model": "VQC",
                "Feature Selection": config["name"],
                "k": config["k"],
                "Fold": fold,
                "Number of Features": n_features,
                "Qubits": n_features,
                "Circuit Depth": circuit_depth,
                "Training Time (s)": train_time,
                "Prediction Time (s)": predict_time,
            }
            row.update(metrics)
            results.append(row)
            print(
                f"    Fold {fold}: Accuracy={metrics['Accuracy']:.4f}, "
                f"F1={metrics['F1']:.4f}, AUC={metrics['ROC-AUC']:.4f}; "
                f"train={train_time:.1f}s"
            )

    results_df, summary_df = save_results(results, "vqc_results.csv")

    vqc_summary = (
        results_df.groupby(["Feature Selection", "k"])[
            ["Number of Features", "Qubits", "Circuit Depth",
             "Training Time (s)", "Prediction Time (s)"]
        ].mean().reset_index()
    )
    vqc_summary.to_csv("vqc_resource_summary.csv", index=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=results_df, x="Feature Selection", y="Accuracy",
        errorbar="sd"
    )
    plt.title("VQC: Diabetes Classification Accuracy")
    plt.xlabel("Feature Selection Method")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("vqc_accuracy.png", dpi=200)
    plt.show()

    print("\nVQC resource summary:")
    print(vqc_summary.to_string(index=False))


if __name__ == "__main__":
    main()
