# Quantum Machine Learning for Breast Cancer Classification

<p align="center">
  <strong>Evaluating Quantum Kernel Learning for Breast Cancer Classification Under Feature Compression</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Healthcare%20AI-blue" />
  <img src="https://img.shields.io/badge/Quantum%20Computing-Qiskit-purple" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" />
  <img src="https://img.shields.io/badge/Language-Python-yellow" />
</p>

## Overview

Breast cancer is one of the most common cancers worldwide. Early and accurate diagnosis plays an important role in improving patient outcomes.

This project investigates the application of **Quantum Machine Learning (QML)** to breast cancer classification using the Breast Cancer Wisconsin Diagnostic (WDBC) dataset.

The primary objective is to evaluate whether quantum kernel learning can provide competitive classification performance compared with classical machine learning models when healthcare data are compressed into a limited number of features.

The project combines classical machine learning, quantum feature maps, and quantum kernel methods using Python and Qiskit.

> **Research focus:** Evaluating predictive performance, computational requirements, circuit complexity, and scalability of quantum machine learning models for healthcare classification.

## Research Question

Does quantum kernel learning provide competitive breast cancer classification performance when the number of input features and qubits is limited?

### Research Objectives

- Develop classical machine learning baselines for breast cancer classification.
- Implement a Quantum Kernel Support Vector Classifier using Qiskit.
- Investigate the impact of PCA-based feature compression.
- Compare classical and quantum models across different feature dimensions.
- Evaluate classification performance and computational requirements.
- Analyze the limitations and scalability of quantum kernel learning for healthcare applications.

## Dataset

This project uses the **Breast Cancer Wisconsin Diagnostic (WDBC)** dataset.

**Dataset source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

| Property | Description |
|---|---|
| Dataset | Breast Cancer Wisconsin Diagnostic |
| Samples | 569 |
| Original features | 30 |
| Target classes | Malignant and Benign |
| Data type | Numerical |
| Missing values | None |
| Application | Breast cancer diagnosis support |

The dataset contains measurements of cell nuclei extracted from digitized fine-needle aspirate images of breast masses.

The features include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry

Each measurement is represented using mean, standard error, and worst-value statistics.

The classification target is:

- **Malignant (M):** Cancerous tumor
- **Benign (B):** Non-cancerous tumor

## Methodology

The project follows a comparative experimental framework.

### 1. Data Preprocessing

- Load the WDBC dataset.
- Separate the diagnostic features and target labels.
- Encode the target labels.
- Perform stratified train-test splitting.
- Standardize the features.
- Apply Principal Component Analysis (PCA) for feature compression.

All preprocessing transformations are fitted using training data only to prevent data leakage.

### 2. Classical Machine Learning

The following classical models are used as baselines:

| Model | Purpose |
|---|---|
| Logistic Regression | Linear classification baseline |
| Support Vector Machine (RBF) | Nonlinear classification baseline |
| Random Forest | Tree-based classification baseline |

### 3. Quantum Machine Learning

The primary quantum model is a **Quantum Kernel Support Vector Classifier (QSVC)**.

The quantum kernel approach uses a quantum feature map to encode classical healthcare features into quantum states.

The similarity between two quantum states is calculated using a fidelity-based quantum kernel.

The kernel is then used by a classical SVM to perform classification.

The implementation uses:

- Python
- Qiskit
- Qiskit Machine Learning
- Qiskit Aer
- Scikit-learn

### Quantum Feature Map

The initial experiments use a ZZ-based quantum feature map.

The number of retained PCA features determines the number of qubits used in the circuit.

| PCA Features | Qubits |
|---|---|
| 2 | 2 |
| 4 | 4 |
| 6 | 6 |
| 8 | 8 |

The experiments investigate how feature dimensionality and circuit complexity affect predictive performance.

## Experimental Design

The classical and quantum models are evaluated using consistent data partitions.

| Parameter | Configuration |
|---|---|
| Dataset | WDBC |
| Train-test split | 80% / 20% |
| Cross-validation | 5-fold stratified CV |
| Feature reduction | PCA |
| Qubit configurations | 2, 4, 6, 8 |
| Quantum feature map | ZZFeatureMap |
| Quantum backend | Statevector simulator |
| Primary quantum model | Quantum Kernel SVM |

The primary evaluation metrics are:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-score
- ROC-AUC
- Training time
- Prediction time

Additional quantum-specific measurements include:

- Number of qubits
- Circuit depth
- Number of circuit evaluations
- Kernel matrix computation time
- Simulation memory requirements

## Classical ML vs. QML Comparison

The project compares the predictive performance and computational requirements of classical and quantum machine learning models.

| Evaluation Metric | Classical ML | Quantum ML |
|---|---|---|
| Accuracy | Yes | Yes |
| Precision | Yes | Yes |
| Recall | Yes | Yes |
| F1-score | Yes | Yes |
| ROC-AUC | Yes | Yes |
| Training time | Yes | Yes |
| Prediction time | Yes | Yes |
| Number of qubits | N/A | Yes |
| Circuit depth | N/A | Yes |
| Scalability | Evaluated | Evaluated |

The purpose is to understand the trade-offs between classical and quantum approaches rather than assume that QML will outperform classical models.

## Results

Experimental results will be added after the models have been implemented and evaluated.

The following table will be used to summarize the main results.

| Model | Features | Qubits | Accuracy | F1-score | ROC-AUC | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 30 | N/A | TBD | TBD | TBD | TBD |
| RBF-SVM | 30 | N/A | TBD | TBD | TBD | TBD |
| Random Forest | 30 | N/A | TBD | TBD | TBD | TBD |
| Quantum Kernel SVM | 2 | 2 | TBD | TBD | TBD | TBD |
| Quantum Kernel SVM | 4 | 4 | TBD | TBD | TBD | TBD |
| Quantum Kernel SVM | 6 | 6 | TBD | TBD | TBD | TBD |
| Quantum Kernel SVM | 8 | 8 | TBD | TBD | TBD | TBD |

**Note:** The results will be populated using actual experimental measurements. No performance advantage is assumed in advance.

## Repository Structure

```text
healthcare-qml/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── classical_models.py
│   ├── quantum_kernel.py
│   ├── vqc.py
│   └── evaluation.py
│
├── experiments/
│   ├── feature_compression.py
│   └── scalability_analysis.py
│
├── results/
│   ├── metrics.csv
│   └── plots/
│
└── paper/
    └── main.tex
```

## Installation

### Prerequisites

- Python 3.10 or compatible version
- pip
- Git

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-qml.git

cd healthcare-qml
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Experiments

The experiments will be organized into separate scripts.

```bash
python src/preprocessing.py
```

```bash
python src/classical_models.py
```

```bash
python src/quantum_kernel.py
```

```bash
python experiments/feature_compression.py
```

The scripts will generate evaluation metrics and visualizations in the `results/` directory.

## Limitations

This project has several limitations:

1. The WDBC dataset is relatively small and may not represent the diversity of real-world patient populations.
2. PCA may remove clinically relevant information during feature compression.
3. Quantum kernel evaluation can become computationally expensive as the number of training samples increases.
4. Statevector simulation does not demonstrate a practical quantum computational advantage.
5. Quantum circuit noise and hardware limitations may affect performance on real quantum devices.
6. Classification performance alone is insufficient to establish clinical usefulness.

The models developed in this project are intended for research and educational purposes only. They are not medical diagnostic systems and should not be used to make clinical decisions.

## Future Research Directions

Potential extensions include:

- Evaluating additional quantum feature maps.
- Comparing Quantum Kernel SVM with Variational Quantum Classifiers.
- Investigating feature-selection methods for healthcare QML.
- Evaluating performance under reduced training data.
- Studying robustness to noisy quantum circuits.
- Running experiments on IBM Quantum hardware, subject to access.
- Extending the analysis to diabetes, heart disease, and other healthcare datasets.

## Research Report

A short IEEE-format research report will accompany this repository.

The report will cover:

- Background and motivation
- Related work
- Healthcare dataset and preprocessing
- Classical machine learning baselines
- Quantum machine learning methodology
- Experimental results
- Computational complexity and scalability
- Limitations and future research

**Report:** Coming soon.

## References

1. UCI Machine Learning Repository, "Breast Cancer Wisconsin (Diagnostic) Data Set." [https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

2. M. Schuld and N. Killoran, "Quantum Machine Learning in Feature Hilbert Spaces," *Physical Review Letters*, vol. 122, 040504, 2019. [https://doi.org/10.1103/PhysRevLett.122.040504](https://doi.org/10.1103/PhysRevLett.122.040504)

3. V. Havlíček et al., "Supervised Learning with Quantum-Enhanced Feature Spaces," *Nature*, vol. 567, pp. 209–212, 2019. [https://doi.org/10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2)

## Author

**Aishwarya Reddy Sarsan**

Graduate Student — Computer Science

Research Interests:
- Quantum Machine Learning
- Healthcare Artificial Intelligence
- Quantum Computing
- Classical and Quantum Hybrid Algorithms

---

⭐ If you find this project interesting, feel free to explore the repository and follow its progress.
