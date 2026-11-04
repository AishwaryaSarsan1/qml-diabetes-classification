"""Shared utilities for the diabetes classification experiments."""
import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
CV_SPLITS = 2  # Initial testing setting; increase for final experiments.

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/pima-indians-diabetes.data.csv"
)
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
CLINICAL_ZERO_COLUMNS = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]

FEATURE_CONFIGS = [
    {"name": "All Features", "selector": None, "k": 8},
    {"name": "Correlation", "selector": "correlation", "k": 2},
    {"name": "Correlation", "selector": "correlation", "k": 4},
    {"name": "Correlation", "selector": "correlation", "k": 6},
    {"name": "SelectKBest", "selector": "selectkbest", "k": 2},
    {"name": "SelectKBest", "selector": "selectkbest", "k": 4},
    {"name": "SelectKBest", "selector": "selectkbest", "k": 6},
    {"name": "RF Importance", "selector": "rf_importance", "k": 2},
    {"name": "RF Importance", "selector": "rf_importance", "k": 4},
    {"name": "RF Importance", "selector": "rf_importance", "k": 6},
]


def load_dataset():
    """Load dataset and mark documented zero values as missing."""
    df = pd.read_csv(DATA_URL, names=COLUMNS)
    X = df.drop(columns=["Outcome"]).copy()
    y = df["Outcome"].copy()
    X[CLINICAL_ZERO_COLUMNS] = X[CLINICAL_ZERO_COLUMNS].replace(0, np.nan)
    return X, y


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Select k features with the largest absolute Pearson correlation to y."""

    def __init__(self, k=2):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        scores = []
        for i in range(X.shape[1]):
            feature = X[:, i]
            if np.std(feature) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(feature, y)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            scores.append(abs(corr))
        self.scores_ = np.asarray(scores)
        self.selected_indices_ = np.argsort(self.scores_)[::-1][:self.k]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_indices_]


class RandomForestImportanceSelector(BaseEstimator, TransformerMixin):
    """Select k features based on a training-fold random forest."""

    def __init__(self, k=2, random_state=RANDOM_STATE):
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
        self.selected_indices_ = np.argsort(self.importances_)[::-1][:self.k]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_indices_]


def preprocess_data(X_train, y_train, X_test, selector_type, k):
    """Fit imputation and feature selection on training data only."""
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    selector = None
    if selector_type == "correlation":
        selector = CorrelationSelector(k=k)
    elif selector_type == "selectkbest":
        selector = SelectKBest(score_func=f_classif, k=k)
    elif selector_type == "rf_importance":
        selector = RandomForestImportanceSelector(
            k=k, random_state=RANDOM_STATE
        )

    if selector is not None:
        selector.fit(X_train_imp, y_train)
        X_train_selected = selector.transform(X_train_imp)
        X_test_selected = selector.transform(X_test_imp)
    else:
        X_train_selected = X_train_imp
        X_test_selected = X_test_imp

    return X_train_selected, X_test_selected


def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def save_results(results, filename):
    """Save fold-level results and grouped mean/std summary."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)

    metric_columns = [
        "Accuracy", "Precision", "Recall", "Specificity", "F1",
        "ROC-AUC", "Training Time (s)", "Prediction Time (s)"
    ]
    summary_df = (
        results_df.groupby(["Model", "Feature Selection", "k"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_path = filename.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved fold-level results: {filename}")
    print(f"Saved summary: {summary_path}")
    return results_df, summary_df
