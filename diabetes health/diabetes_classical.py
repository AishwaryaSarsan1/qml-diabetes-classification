"""Run classical ML models on the Pima diabetes dataset.

Run:
    python diabetes_classical.py
"""
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from diabetes_common import (
    RANDOM_STATE, CV_SPLITS, FEATURE_CONFIGS, load_dataset,
    preprocess_data, calculate_metrics, save_results
)


def create_classical_model(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE
        )
    if model_name == "RBF-SVM":
        return SVC(
            kernel="rbf", probability=True, class_weight="balanced",
            random_state=RANDOM_STATE
        )
    raise ValueError(f"Unknown model: {model_name}")


def main():
    X, y = load_dataset()
    print("Dataset shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())

    models = ["Logistic Regression", "Random Forest", "RBF-SVM"]
    cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    results = []

    for config in FEATURE_CONFIGS:
        print(f"\nFeature selection: {config['name']}, k={config['k']}")
        for model_name in models:
            print(f"  Model: {model_name}")
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                X_train_sel, X_test_sel = preprocess_data(
                    X_train, y_train, X_test,
                    config["selector"], config["k"]
                )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_sel)
                X_test_scaled = scaler.transform(X_test_sel)

                model = create_classical_model(model_name)
                start = time.time()
                model.fit(X_train_scaled, y_train)
                train_time = time.time() - start

                start = time.time()
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                predict_time = time.time() - start

                metrics = calculate_metrics(y_test, y_pred, y_prob)
                row = {
                    "Model": model_name,
                    "Feature Selection": config["name"],
                    "k": config["k"],
                    "Fold": fold,
                    "Number of Features": X_train_sel.shape[1],
                    "Qubits": 0,
                    "Circuit Depth": 0,
                    "Training Time (s)": train_time,
                    "Prediction Time (s)": predict_time,
                }
                row.update(metrics)
                results.append(row)
                print(
                    f"    Fold {fold}: Accuracy={metrics['Accuracy']:.4f}, "
                    f"F1={metrics['F1']:.4f}, AUC={metrics['ROC-AUC']:.4f}"
                )

    results_df, summary_df = save_results(
        results, "classical_results.csv"
    )

    # Plot fold-level accuracy by method/model.
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=results_df, x="Feature Selection", y="Accuracy",
        hue="Model", errorbar="sd"
    )
    plt.title("Classical Models: Diabetes Classification Accuracy")
    plt.xlabel("Feature Selection Method")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("classical_accuracy.png", dpi=200)
    plt.show()

    print("\nBest mean configurations:")
    mean_results = (
        results_df.groupby(["Model", "Feature Selection", "k"])[
            ["Accuracy", "F1", "ROC-AUC", "Recall"]
        ].mean().reset_index()
    )
    for metric in ["Accuracy", "F1", "ROC-AUC", "Recall"]:
        best = mean_results.loc[mean_results[metric].idxmax()]
        print(f"\nBest by {metric}:\n{best}")


if __name__ == "__main__":
    main()
