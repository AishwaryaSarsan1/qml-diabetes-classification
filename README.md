# Evaluating Quantum Machine Learning for Diabetes Risk Prediction Under Different Feature-Selection Strategies

## Overview

This project investigates the use of Quantum Machine Learning (QML) for diabetes classification using clinical measurements from the Pima Indians Diabetes Dataset.

The primary goal is to evaluate how different feature-selection strategies affect the performance and computational cost of a Variational Quantum Classifier (VQC). The quantum model will be compared with classical machine-learning models under consistent experimental conditions.

This project is intended for academic research and educational purposes. It is not a clinical diagnostic tool.

## Research Question

**How do different feature-selection strategies affect the predictive performance and computational cost of a Variational Quantum Classifier for diabetes classification compared with classical machine-learning models?**

The project explores whether reducing the number of input features can improve the feasibility of quantum model training while maintaining competitive classification performance.

## Dataset

The project uses the **Pima Indians Diabetes Dataset**, a commonly studied dataset for binary diabetes classification.

The dataset contains clinical measurements for 768 female patients of Pima Indian heritage who were at least 21 years old.

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skinfold thickness |
| Insulin | Two-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |

### Target

The target variable, `Outcome`, represents the recorded diabetes classification:

- `0`: No diabetes recorded
- `1`: Diabetes recorded

**Dataset considerations:** Several clinical measurements contain zero values that may represent missing or invalid measurements rather than true physiological values. These values will be investigated and handled during preprocessing.

The dataset also has population-specific limitations. Results should not be interpreted as validated diabetes-risk predictions for the general population.

## Methodology

The project follows a controlled experimental workflow:

1. Load and explore the dataset.
2. Inspect missing, invalid, and zero-valued measurements.
3. Split the data into stratified training and test sets.
4. Fit preprocessing transformations using only the training data.
5. Apply different feature-selection strategies.
6. Train classical machine-learning models.
7. Train a Variational Quantum Classifier using selected features.
8. Evaluate predictive performance and computational cost.
9. Compare results across feature subsets and model types.

Feature selection and preprocessing will be performed within the training data to prevent data leakage.

## Models

### Classical Machine-Learning Models

The following models will serve as classical baselines:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM) with an RBF kernel

### Quantum Machine-Learning Model

**Variational Quantum Classifier (VQC)**

The VQC uses a parameterized quantum circuit to encode selected features and learn a classification function.

The circuit consists of:

- A feature-encoding circuit
- Parameterized quantum gates
- Measurement operations
- A classical optimization procedure for updating circuit parameters

The number of selected features will be connected to the number of qubits used in the quantum model.

The initial experiments will use small circuit configurations, such as 2 and 4 qubits, to keep simulation computationally manageable.

## Feature-Selection Strategies

The project will compare the following strategies:

| Strategy | Description |
|---|---|
| All Features | Use all eight available clinical features |
| Correlation-Based Selection | Select features based on their relationship with the target |
| SelectKBest | Select the top features using a statistical scoring method |
| Random Forest Feature Importance | Select features based on importance estimated by a Random Forest |

The selected features will be evaluated using both classical models and the VQC where appropriate.

## Evaluation Metrics

Models will be evaluated using:

- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1-score
- ROC-AUC
- Training time
- Prediction time

Because diabetes classification involves potential class imbalance, accuracy will not be used as the sole performance measure.

## Experimental Design

The experiments will investigate:

- The effect of feature selection on classical model performance
- The effect of feature selection on VQC performance
- The relationship between selected feature count and qubit count
- The computational cost of training quantum and classical models
- Whether reducing the feature dimension changes the performance trade-off

Where feasible, experiments will be repeated with multiple random seeds to assess result stability.

## Repository Structure

```text
healthcare-qml/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── classical_models.py
│   ├── vqc_model.py
│   └── evaluation.py
├── experiments/
│   └── feature_selection.py
├── results/
│   ├── metrics.csv
│   └── plots/
└── paper/
    └── main.tex
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-qml.git
cd healthcare-qml
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

The experiment scripts will be added as the project develops.

The planned workflow is:

```bash
python experiments/feature_selection.py
```

Results will be saved in the `results/` directory.

## Results

Experimental results will be added after the models have been trained and evaluated.

| Model | Feature-Selection Strategy | Number of Features | Accuracy | F1-score | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Logistic Regression | All Features | 8 | TBD | TBD | TBD |
| Random Forest | All Features | 8 | TBD | TBD | TBD |
| RBF-SVM | All Features | 8 | TBD | TBD | TBD |
| VQC | Selected Features | TBD | TBD | TBD | TBD |

No performance claims are made before completing the experiments.

## Limitations

- The dataset is relatively small and population-specific.
- The dataset may contain invalid or missing clinical measurements.
- Quantum experiments may rely on classical simulation rather than physical quantum hardware.
- VQC performance can depend on circuit architecture, optimizer, initialization, and measurement variability.
- A simulator-based comparison does not demonstrate practical quantum advantage.
- The dataset is not intended to establish a clinical diagnosis or provide general-purpose diabetes-risk predictions.

## Future Work

Potential extensions include:

- Testing additional feature-selection methods
- Exploring different quantum circuit architectures
- Comparing VQC with quantum kernel methods
- Evaluating model stability across repeated splits
- Testing on additional healthcare datasets
- Running experiments on available quantum hardware

## References

1. Pima Indians Diabetes Dataset, UCI Machine Learning Repository.
2. Qiskit documentation: https://qiskit.org/documentation/
3. Qiskit Machine Learning documentation: https://qiskit-community.github.io/qiskit-machine-learning/
4. Scikit-learn documentation: https://scikit-learn.org/

## Disclaimer

This project is for educational and research purposes only. It does not provide medical advice, diagnosis, or treatment recommendations. The results should not be used to make healthcare decisions.
