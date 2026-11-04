# ============================================================
# Diabetes Classification Using Classical ML and a VQC
# ============================================================

import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Create a folder to save all graphs
GRAPH_DIR = Path.cwd() / "diabetes_graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

print("Graphs will be saved in:", GRAPH_DIR.resolve())
import seaborn as sns
from IPython.display import display
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import VQC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# Set lower values for initial testing.
# Increase CV_SPLITS and VQC_MAXITER for final experiments.
CV_SPLITS = 2
VQC_MAXITER = 5

print("Imports completed.")

# ============================================================
# Load Pima Indians Diabetes Dataset
# ============================================================

url = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/pima-indians-diabetes.data.csv"
)

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv(url, names=columns)

print("Dataset shape:", df.shape)
display(df.head())

print("\nTarget distribution:")
print(df["Outcome"].value_counts())

print("\nDataset summary:")
display(df.describe())
# ============================================================
# Prepare features and target
# ============================================================

X = df.drop(columns=["Outcome"]).copy()
y = df["Outcome"].copy()

clinical_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[clinical_columns] = X[clinical_columns].replace(0, np.nan)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nMissing values:")
display(X.isnull().sum())
# ============================================================
# Custom correlation-based feature selector
# ============================================================

class CorrelationSelector(BaseEstimator, TransformerMixin):

    def __init__(self, k=2):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        correlations = []

        for i in range(X.shape[1]):
            feature = X[:, i]

            if np.std(feature) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(feature, y)[0, 1]

                if np.isnan(corr):
                    corr = 0.0

            correlations.append(abs(corr))

        self.scores_ = np.asarray(correlations)

        self.selected_indices_ = np.argsort(
            self.scores_
        )[::-1][:self.k]

        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.selected_indices_]


# ============================================================
# Random Forest importance-based feature selector
# ============================================================

class RandomForestImportanceSelector(
    BaseEstimator,
    TransformerMixin
):

    def __init__(self, k=2, random_state=42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)

        self.rf_model_ = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            class_weight="balanced"
        )

        self.rf_model_.fit(X, y)

        self.importances_ = self.rf_model_.feature_importances_

        self.selected_indices_ = np.argsort(
            self.importances_
        )[::-1][:self.k]

        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.selected_indices_]
# ============================================================
# Feature-selection configurations
# ============================================================

feature_configs = [
    {
        "name": "All Features",
        "selector": None,
        "k": 8
    },

    {
        "name": "Correlation",
        "selector": "correlation",
        "k": 2
    },
    {
        "name": "Correlation",
        "selector": "correlation",
        "k": 4
    },
    {
        "name": "Correlation",
        "selector": "correlation",
        "k": 6
    },

    {
        "name": "SelectKBest",
        "selector": "selectkbest",
        "k": 2
    },
    {
        "name": "SelectKBest",
        "selector": "selectkbest",
        "k": 4
    },
    {
        "name": "SelectKBest",
        "selector": "selectkbest",
        "k": 6
    },

    {
        "name": "RF Importance",
        "selector": "rf_importance",
        "k": 2
    },
    {
        "name": "RF Importance",
        "selector": "rf_importance",
        "k": 4
    },
    {
        "name": "RF Importance",
        "selector": "rf_importance",
        "k": 6
    }
]

print("Number of feature configurations:", len(feature_configs))
# ============================================================
# Classical machine learning models
# ============================================================

def create_classical_model(model_name):

    if model_name == "Logistic Regression":

        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    elif model_name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    elif model_name == "RBF-SVM":

        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")


classical_models = [
    "Logistic Regression",
    "Random Forest",
    "RBF-SVM"
]
# ============================================================
# Create Variational Quantum Classifier
# ============================================================

def create_vqc(num_features):

    feature_map = ZZFeatureMap(
        feature_dimension=num_features,
        reps=1
    )

    ansatz = RealAmplitudes(
        num_qubits=num_features,
        reps=1
    )

    sampler = StatevectorSampler(
        default_shots=1024,
        seed=RANDOM_STATE
    )

    optimizer = COBYLA(
        maxiter=VQC_MAXITER
    )

    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler
    )

    return model
# ============================================================
# Fit preprocessing and feature selection on training data
# ============================================================

def preprocess_data(X_train, y_train, X_test, selector_type, k):

    # Impute using only training data
    imputer = SimpleImputer(strategy="median")

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Feature selection
    selector = None

    if selector_type == "correlation":

        selector = CorrelationSelector(k=k)

    elif selector_type == "selectkbest":

        selector = SelectKBest(
            score_func=f_classif,
            k=k
        )

    elif selector_type == "rf_importance":

        selector = RandomForestImportanceSelector(
            k=k,
            random_state=RANDOM_STATE
        )

    if selector is not None:

        selector.fit(X_train_imp, y_train)

        X_train_selected = selector.transform(X_train_imp)
        X_test_selected = selector.transform(X_test_imp)

    else:

        X_train_selected = X_train_imp
        X_test_selected = X_test_imp

    return X_train_selected, X_test_selected
# ============================================================
# Calculate evaluation metrics
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Specificity": specificity,

        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            y_prob
        )
    }
# ============================================================
# Run experiments
# ============================================================

all_results = []

cv = StratifiedKFold(
    n_splits=CV_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

model_names = classical_models + ["VQC"]

for config in feature_configs:

    selector_type = config["selector"]
    k = config["k"]
    feature_name = config["name"]

    print(
        f"\nFeature selection: {feature_name}, "
        f"k={k}"
    )

    for model_name in model_names:

        print(f"  Model: {model_name}")

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(X, y),
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # --------------------------------------------
            # Preprocessing and feature selection
            # --------------------------------------------

            X_train_selected, X_test_selected = preprocess_data(
                X_train,
                y_train,
                X_test,
                selector_type,
                k
            )

            number_of_features = X_train_selected.shape[1]

            # --------------------------------------------
            # Classical models
            # --------------------------------------------

            if model_name != "VQC":

                scaler = StandardScaler()

                X_train_scaled = scaler.fit_transform(
                    X_train_selected
                )

                X_test_scaled = scaler.transform(
                    X_test_selected
                )

                model = create_classical_model(model_name)

                start_train = time.time()

                model.fit(
                    X_train_scaled,
                    y_train
                )

                train_time = time.time() - start_train

                start_predict = time.time()

                y_pred = model.predict(
                    X_test_scaled
                )

                y_prob = model.predict_proba(
                    X_test_scaled
                )[:, 1]

                predict_time = time.time() - start_predict

                qubits = 0
                circuit_depth = 0

            # --------------------------------------------
            # Variational Quantum Classifier
            # --------------------------------------------

            else:

                # Map selected classical features to [0, pi]
                scaler = MinMaxScaler(
                    feature_range=(0, np.pi)
                )

                X_train_scaled = scaler.fit_transform(
                    X_train_selected
                )

                X_test_scaled = scaler.transform(
                    X_test_selected
                )

                model = create_vqc(
                    num_features=number_of_features
                )

                start_train = time.time()

                model.fit(
                    X_train_scaled,
                    y_train.to_numpy()
                )

                train_time = time.time() - start_train

                start_predict = time.time()

                y_pred = np.asarray(
                    model.predict(X_test_scaled)
                ).astype(int).ravel()

                # Qiskit VQC normally provides predict_proba.
                # Handle both two-column and one-column outputs.
                y_prob_all = np.asarray(
                    model.predict_proba(X_test_scaled)
                )

                if y_prob_all.ndim == 2 and y_prob_all.shape[1] == 2:

                    y_prob = y_prob_all[:, 1]

                elif y_prob_all.ndim == 2 and y_prob_all.shape[1] == 1:

                    y_prob = y_prob_all[:, 0]

                else:

                    y_prob = y_prob_all.ravel()

                predict_time = time.time() - start_predict

                qubits = number_of_features

                circuit_depth = (
                    model.feature_map.depth()
                    + model.ansatz.depth()
                )

            # --------------------------------------------
            # Metrics
            # --------------------------------------------

            metrics = calculate_metrics(
                y_test,
                y_pred,
                y_prob
            )

            result = {
                "Model": model_name,
                "Feature Selection": feature_name,
                "k": k,
                "Fold": fold,
                "Number of Features": number_of_features,
                "Qubits": qubits,
                "Circuit Depth": circuit_depth,
                "Training Time (s)": train_time,
                "Prediction Time (s)": predict_time
            }

            result.update(metrics)

            all_results.append(result)

            print(
                f"    Fold {fold}: "
                f"Accuracy={metrics['Accuracy']:.4f}, "
                f"F1={metrics['F1']:.4f}, "
                f"AUC={metrics['ROC-AUC']:.4f}"
            )

results_df = pd.DataFrame(all_results)

print("\nExperiment completed.")
display(results_df.head())
# ============================================================
# Save raw fold-level results
# ============================================================

results_df.to_csv(
    "diabetes_qml_fold_results.csv",
    index=False
)

print("Saved: diabetes_qml_fold_results.csv")
# ============================================================
# Aggregate cross-validation results
# ============================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "ROC-AUC",
    "Training Time (s)",
    "Prediction Time (s)"
]

summary_df = (
    results_df
    .groupby(
        ["Model", "Feature Selection", "k"]
    )[metric_columns]
    .agg(["mean", "std"])
    .reset_index()
)

summary_df.to_csv(
    "diabetes_qml_summary.csv",
    index=False
)

display(summary_df)

print("Saved: diabetes_qml_summary.csv")
# ============================================================
# Accuracy comparison
# ============================================================

plt.figure(figsize=(14, 7))

sns.barplot(
    data=results_df,
    x="Feature Selection",
    y="Accuracy",
    hue="Model",
    errorbar="sd"
)

plt.title("Diabetes Classification Accuracy")
plt.xlabel("Feature Selection Method")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.legend(
    title="Model",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "diabetes_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()
# ============================================================
# Accuracy vs number of selected features
# ============================================================

plt.figure(figsize=(12, 7))

sns.lineplot(
    data=results_df,
    x="k",
    y="Accuracy",
    hue="Model",
    style="Feature Selection",
    markers=True,
    errorbar="sd"
)

plt.title("Accuracy vs Number of Selected Features")
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "accuracy_vs_features.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()
# ============================================================
# VQC resource analysis
# ============================================================

vqc_results = results_df[
    results_df["Model"] == "VQC"
].copy()

vqc_summary = (
    vqc_results
    .groupby(
        ["Feature Selection", "k"]
    )[
        [
            "Number of Features",
            "Qubits",
            "Circuit Depth",
            "Training Time (s)",
            "Prediction Time (s)"
        ]
    ]
    .mean()
    .reset_index()
)

display(vqc_summary)

vqc_summary.to_csv(
    "vqc_resource_summary.csv",
    index=False
)
# ============================================================
# VQC qubit count
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=vqc_summary,
    x="Feature Selection",
    y="Qubits"
)

plt.title("VQC Qubit Count by Feature Selection Method")
plt.xlabel("Feature Selection")
plt.ylabel("Number of Qubits")

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "vqc_qubit_count.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()
# ============================================================
# VQC circuit depth
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=vqc_summary,
    x="Feature Selection",
    y="Circuit Depth"
)

plt.title("VQC Circuit Depth by Feature Selection Method")
plt.xlabel("Feature Selection")
plt.ylabel("Circuit Depth")

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "vqc_circuit_depth.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()
# ============================================================
# Find the best-performing configurations
# ============================================================

mean_results = (
    results_df
    .groupby(
        ["Model", "Feature Selection", "k"]
    )[metric_columns]
    .mean()
    .reset_index()
)

for metric in ["Accuracy", "F1", "ROC-AUC", "Recall"]:

    best_row = mean_results.loc[
        mean_results[metric].idxmax()
    ]

    print(f"\nBest configuration by {metric}:")
    print(best_row)