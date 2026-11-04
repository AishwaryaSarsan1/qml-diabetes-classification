
# 🩺 Evaluating Quantum Machine Learning for Diabetes Risk Prediction

**A comparative study of classical machine learning and variational
quantum classification for diabetes classification.**
:::

------------------------------------------------------------------------

## 🎯 Overview

This project proposes an investigation of **Quantum Machine Learning
(QML)** for diabetes classification using clinical measurements from the
**Pima Indians Diabetes Dataset**.

The primary goal is to evaluate how different feature-selection
strategies affect the predictive performance and computational cost of a
**Variational Quantum Classifier (VQC)**. The quantum model will be
compared with classical machine-learning models under consistent
experimental conditions.

The project will explore whether reducing the number of input features
can make quantum model training more computationally manageable while
maintaining useful classification performance.

> **Project type:** Academic research proposal\
> **Status:** Proposed work; experimental results will be added after
> evaluation.

## ❓ Research Question

**How do different feature-selection strategies affect the predictive
performance and computational cost of a Variational Quantum Classifier
for diabetes classification compared with classical machine-learning
models?**

The study will examine the relationship between feature dimensionality,
classification performance, and quantum resource requirements.

## 📊 Dataset

The project will use the **Pima Indians Diabetes Dataset**, a commonly
studied dataset for binary diabetes classification.

The dataset contains **768 observations** of female patients of Pima
Indian heritage who were at least 21 years old.

### Input Features

  Feature                      Description
  ---------------------------- ------------------------------
  `Pregnancies`                Number of pregnancies
  `Glucose`                    Plasma glucose concentration
  `BloodPressure`              Diastolic blood pressure
  `SkinThickness`              Triceps skinfold thickness
  `Insulin`                    Two-hour serum insulin
  `BMI`                        Body mass index
  `DiabetesPedigreeFunction`   Diabetes pedigree function
  `Age`                        Age in years

### Target Variable

The target column is `Outcome`:

    Value Meaning
  ------- ----------------------
      `0` No diabetes recorded
      `1` Diabetes recorded

**Data consideration:** Zero values in some clinical measurements may
represent missing or invalid measurements rather than true physiological
values. These values will be investigated and handled during
preprocessing.

The dataset is population-specific and has limitations. Results will not
be interpreted as validated diabetes-risk predictions for the general
population.

## 🔬 Proposed Methodology

The project will follow this workflow:

1.  Load and explore the dataset.
2.  Inspect missing, invalid, and zero-valued measurements.
3.  Create stratified training and evaluation splits.
4.  Fit preprocessing transformations using training data only.
5.  Apply the selected feature-selection strategy.
6.  Train the classical machine-learning models.
7.  Train the Variational Quantum Classifier using selected features.
8.  Evaluate predictive performance and computational cost.
9.  Compare the results across feature subsets and model types.

To prevent **data leakage**, preprocessing and feature selection will be
fitted only on the training portion of each split and then applied to
its corresponding evaluation data.

## 🤖 Models

### Classical Machine-Learning Models

The following models will be used as classical baselines:

  -----------------------------------------------------------------------
  Model                               Purpose
  ----------------------------------- -----------------------------------
  Logistic Regression                 Linear classification baseline

  Random Forest                       Tree-ensemble classification

  RBF-SVM                             Nonlinear classification using a
                                      radial basis function kernel
  -----------------------------------------------------------------------

### Quantum Machine-Learning Model

**Variational Quantum Classifier (VQC)**

The VQC will use a parameterized quantum circuit to encode selected
input features and learn a classification function.

The circuit is planned to contain:

-   **Feature encoding:** maps classical input features into a quantum
    state.
-   **Parameterized quantum gates:** provide trainable circuit
    parameters.
-   **Measurement operations:** produce values used for classification.
-   **Classical optimizer:** updates circuit parameters during training.

The number of selected features will be connected to the number of
qubits used by the quantum model. Initial experiments will focus on
small circuit configurations, such as 2 and 4 qubits, to keep simulation
computationally manageable.

## 🧩 Feature-Selection Strategies

The study will compare four feature-selection conditions:

  -----------------------------------------------------------------------
  Strategy                            Description
  ----------------------------------- -----------------------------------
  **All Features**                    Uses all eight available clinical
                                      features

  **Correlation-Based Selection**     Selects features based on their
                                      relationship with the target

  **SelectKBest**                     Selects top features using a
                                      statistical scoring method

  **Random Forest Feature             Selects features based on
  Importance**                        importance estimated by a Random
                                      Forest
  -----------------------------------------------------------------------

Different feature counts will be investigated to examine how
dimensionality reduction affects classical and quantum model
performance.

## 📈 Evaluation Metrics

Models will be evaluated using:

  Metric                 Purpose
  ---------------------- ----------------------------------------------------------
  Accuracy               Overall proportion of correct predictions
  Precision              Proportion of predicted positive cases that are positive
  Recall / Sensitivity   Ability to identify positive cases
  Specificity            Ability to identify negative cases
  F1-score               Harmonic mean of precision and recall
  ROC-AUC                Ability to distinguish between the two classes
  Training Time          Time required to train a model
  Prediction Time        Time required to generate predictions

For the VQC, the **number of qubits** and **circuit depth** will also be
recorded to help assess quantum resource requirements.

Because the dataset may involve class imbalance, accuracy will not be
treated as the only measure of model performance.

## 🧪 Experimental Design

The experiments will investigate:

-   How feature selection affects classical model performance.
-   How feature selection affects VQC performance.
-   The relationship between selected feature count and qubit count.
-   The computational cost of training quantum and classical models.
-   Whether reducing feature dimensions changes the trade-off between
    predictive performance and computational cost.

Where feasible, experiments will be repeated using multiple random seeds
to assess result stability.



## ⚙️ Installation

### 1. Clone the repository

Replace `YOUR_USERNAME` with your GitHub username:

``` bash
git clone https://github.com/YOUR_USERNAME/healthcare-qml.git
cd healthcare-qml
```

### 2. Create a virtual environment

``` bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

``` bash
source .venv/bin/activate
```

**Windows**

``` bash
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

The project is expected to use Python, Qiskit, Qiskit Machine Learning,
scikit-learn, NumPy, and pandas.

## ▶️ Running the Experiments

The experiment scripts will be developed as part of the project.

The planned command is:

``` bash
python experiments/feature_selection.py
```

Generated metrics and plots will be saved in the `results/` directory.

## 📋 Expected Results

The experiments are intended to provide a comparative evaluation of
classical and quantum classification models under different
feature-selection strategies.

The following table will be completed after the experiments are
conducted:

  ------------------------------------------------------------------------------------
  Model        Feature-Selection      Number of     Accuracy     F1-score      ROC-AUC
               Strategy                Features                           
  ------------ ------------------- ------------ ------------ ------------ ------------
  Logistic     All Features                   8          TBD          TBD          TBD
  Regression                                                              

  Random       All Features                   8          TBD          TBD          TBD
  Forest                                                                  

  RBF-SVM      All Features                   8          TBD          TBD          TBD

  VQC          Selected Features            TBD          TBD          TBD          TBD
  ------------------------------------------------------------------------------------

No performance claims are made before the experiments are completed.

## ⚠️ Limitations

-   The dataset is relatively small and population-specific.
-   Some clinical measurements may contain missing or invalid values.
-   Quantum experiments may use classical simulation rather than
    physical quantum hardware.
-   VQC performance may depend on circuit architecture, optimizer,
    parameter initialization, and measurement variability.
-   A simulator-based comparison does not demonstrate practical quantum
    advantage.
-   The dataset is not intended to establish a clinical diagnosis or
    provide general-purpose diabetes-risk predictions.

## 🚀 Future Work

Potential extensions include:

-   Testing additional feature-selection methods.
-   Exploring different quantum circuit architectures.
-   Comparing VQC with quantum kernel methods.
-   Evaluating model stability across repeated data splits.
-   Testing on additional healthcare datasets.
-   Running experiments on available quantum hardware.

## 📚 References

1.  **Pima Indians Diabetes Dataset**, UCI Machine Learning Repository.
2.  **Qiskit Documentation:** https://qiskit.org/documentation/
3.  **Qiskit Machine Learning Documentation:**
    https://qiskit-community.github.io/qiskit-machine-learning/
4.  **Scikit-learn Documentation:** https://scikit-learn.org/


------------------------------------------------------------------------

::: {align="center"}
**Academic Research Project · Quantum Machine Learning · Healthcare
Classification**
:::
