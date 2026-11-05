# Diabetes Healthcare QML
A research-oriented Python project exploring Quantum Machine Learning (QML) for diabetes classification. The project compares a Qiskit Variational Quantum Classifier (VQC) with classical machine learning models under different feature-selection strategies.
## Table of Contents
- 1. Project Title and Description
- 2. Prerequisites
- 3. Installation Instructions
- 4. Execution Instructions
- 5. Input/Output Explanation
- 6. Features
- 7. Experimental Results
- 8. Troubleshooting
- 9. Acknowledgements
- 10. AI Assistance Disclaimer
## 1. Project Title and Description
### Project Title
Evaluating Quantum Machine Learning for Diabetes Risk Prediction Under Different Feature-Selection Strategies
### Brief Overview
This project explores Quantum Machine Learning for a healthcare classification problem: predicting diabetes status from patient health measurements.

The project aims to:
1. Compare classical machine learning and quantum machine learning for diabetes classification.
2. Evaluate the effect of different feature-selection strategies.
3. Study how the number of selected features affects VQC performance and computational cost.
4. Report classification metrics, training and prediction time, qubit requirements, circuit depth, scalability, and limitations.
Healthcare Dataset
The experiments use the Pima Indians Diabetes Dataset.
- Samples: 768 patient records
- Predictor features: 8
- Target: Outcome (0 = non-diabetic, 1 = diabetic)
- Class distribution: 500 records in class 0 and 268 records in class 1
The predictor variables include health measurements such as glucose, blood pressure, skin thickness, insulin, BMI, age, and pregnancies.

Research Question
How do different feature-selection strategies affect the predictive performance and computational requirements of a Variational Quantum Classifier compared with classical machine learning models for diabetes classification?
Technology Stack
- Python
- Qiskit and Qiskit Machine Learning
- scikit-learn
- pandas and NumPy
- Matplotlib and Seaborn
- Weights & Biases (optional experiment tracking)
## 2. Prerequisites
Software Requirements
Component	Purpose
Python 3.10 or compatible environment	Running experiments in Jupyter
Qiskit	Quantum circuit implementation
Qiskit Machine Learning	Variational Quantum Classifier
NumPy	Numerical operations
pandas	Dataset loading and processing
scikit-learn	Classical models, preprocessing, and metrics
Matplotlib	Result plots
Seaborn	Statistical visualizations
Weights & Biases (optional)	Experiment tracking
Jupyter Notebook (optional)	Exploring results


Use the compatible package versions listed in requirements.txt, if provided.
Hardware Requirements
1. CPU: Modern multi-core processor
2. RAM: 8 GB recommended
3. Disk space: Approximately 1 GB for the environment, dataset, and outputs
4. Operating system: Windows, Linux, or macOS
5. Quantum hardware: Not required for the simulator-based experiments
The current VQC experiments use Qiskit's simulator workflow. Simulator results do not establish an advantage on physical quantum hardware.

## 3. Installation Instructions
Step 1: Clone or Download the Project
Clone the repository:
git clone YOUR_GITHUB_REPOSITORY_URL
cd diabetes-healthcare-qml
Alternatively, download the repository as a ZIP file and extract it. Replace the folder name above if your repository uses a different name.
Step 2: Create a Virtual Environment
python -m venv .venv
Activate it.

Windows:
.venv\Scripts\activate
macOS/Linux:
source .venv/bin/activate
Step 3: Install Dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Make sure the installed Qiskit and Qiskit Machine Learning versions are compatible.
Step 4: Prepare the Dataset
1. Obtain the Pima Indians Diabetes Dataset from its documented source.
2. Place the CSV file in the location expected by the project scripts.
3. Update the dataset path in the configuration or common utility file if needed.
4. Confirm that the target column is named Outcome, or update the code to match your CSV.
Do not commit private patient information to the repository.
Step 5: (Optional) Configure Weights & Biases
Install and authenticate W&B:
python -m pip install wandb
wandb login
The experiment scripts use these W&B project names:
- Diabetes-Classical
- Diabetes-VQC
Never hard-code or commit your W&B API key. Use the login command or an environment variable.

## 4. Execution Instructions
Step 1: Activate the Environment
Activate the .venv environment created during installation.
Step 2: Run the Experiments
If your repository includes the unified script:
python diabetes_qml.py
For the split implementation, run the scripts separately:
python diabetes_classical.py
python diabetes_vqc.py
The split scripts use shared utilities from diabetes_common.py.

Run only the entry-point scripts that are present in your repository. Check the script configuration for the expected dataset path and output directory.
Step 3: Review the Outputs
Depending on the script configuration, generated outputs may include:
- Fold-level model metrics
- Aggregated performance summaries
- Training and prediction times
- Feature-selection results
- Comparison plots
- W&B runs and artifacts, if enabled
## 5. Input/Output Explanation
### 5.1 Dataset Input and Preprocessing
Input: A CSV dataset containing patient health measurements and the diabetes target label Outcome.

Processing:
1. Load the dataset.
2. Separate predictor variables from the target.
3. Treat zero values in selected clinical measurement columns as missing where specified by the experiment.
4. Apply median imputation using the training fold.
5. Evaluate using stratified cross-validation.
The current configuration uses two-fold stratified cross-validation with random_state=42.
5.2 Feature Selection
The project evaluates:
1. All 8 features
2. Correlation-based feature selection
3. SelectKBest using the ANOVA F-test
4. Random Forest feature importance
Reduced feature configurations use k = 2, 4, and 6.

For the current VQC design, the selected feature count determines the number of input features encoded into the circuit and the number of qubits.
### 5.3 Classical Machine Learning Models
The classical baselines are:
- Logistic Regression
- Random Forest
- Support Vector Machine (RBF kernel)
The current experiment uses balanced class weights.

Outputs:
- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- ROC-AUC
- Training time
- Prediction time
### 5.4 Quantum Machine Learning – VQC
The quantum model uses Qiskit's Variational Quantum Classifier (VQC).

Current configuration:
- Feature map: ZZFeatureMap, 1 repetition
- Ansatz: RealAmplitudes, 1 repetition
- Sampler: StatevectorSampler
- Shots: 1024
- Optimizer: COBYLA
- Maximum optimizer iterations: 5
- Feature scaling: MinMax scaling to [0, π]
Processing flow:
1. Select the features for the experiment.
2. Scale selected features for quantum encoding.
3. Encode data using the ZZ feature map.
4. Apply the parameterized RealAmplitudes ansatz.
5. Train the VQC using the configured optimizer.
6. Predict diabetes classes for validation samples.
7. Calculate classification metrics and computational measurements.
Outputs:
- Accuracy, precision, recall, specificity, F1-score, and ROC-AUC
- Training and prediction time
- Number of selected features/qubits
- Circuit depth, as measured by the experiment code
### 5.5 Experiment Tracking (Optional)
When W&B is enabled, metrics and configuration details are logged to:
- Diabetes-Classical
- Diabetes-VQC
This supports review and comparison of experiment runs.
## 6. Features
Healthcare Classification
- Diabetes classification using patient health measurements
- Binary prediction of diabetes outcome
- Educational research using a public healthcare dataset


Feature-Selection Experiments
- All-feature baseline
- Correlation-based selection
- ANOVA SelectKBest
- Random Forest feature importance
- Comparison across selected feature counts
  
Classical ML Baselines
- Logistic Regression
- Random Forest
- RBF Support Vector Machine
  
Quantum Machine Learning
- Qiskit Variational Quantum Classifier
- ZZFeatureMap data encoding
- RealAmplitudes variational ansatz
- Qubit count based on selected features
  
Evaluation and Reporting
- Classification performance metrics
- Training and prediction time
- Qubit count and circuit depth
- Model and feature-selection comparisons
- Visualizations and optional W&B tracking
## 7. Experimental Results
### 7.1 Best Accuracy by Model
The following values are from the current two-fold experiment results. They are the best observed accuracy configurations for each model in this experiment.
Model	Best Feature Configuration	Accuracy
Logistic Regression	Random Forest Importance, k = 4	0.7461
Random Forest	SelectKBest, k = 4	0.7656
RBF-SVM	Correlation, k = 4	0.7448
VQC	SelectKBest, k = 4	0.5872


### 7.2 Interpretation
- Classical models achieved higher accuracy than the VQC in the current experiment.
- The best observed VQC accuracy was approximately 0.5872.
- VQC performance was weak across several tested configurations.
- These results do not demonstrate a quantum advantage.
- Feature selection affected predictive performance and computational requirements.
7.3 Qubit Requirements and Scalability
In the current design, the number of selected input features determines the qubit count.
Selected Features	Qubits
2	2
4	4
6	6
8	8


The current feature map and ansatz each use one repetition. Report the circuit depth measured by the actual execution code, along with the circuit definition and transpilation settings.

VQC training time increased as the number of selected features increased in the current runs. This highlights a practical computational challenge when scaling variational circuits to larger feature sets.
7.4 Limitations and Future Experiments
The current experiment uses two-fold cross-validation and only five VQC optimizer iterations. Results should therefore be treated as preliminary.

Future experiments should consider:
- More cross-validation folds
- Repeated stratified cross-validation or a separate held-out test set
- More optimizer iterations and optimizer comparisons
- Multiple random seeds
- Repeated timing measurements under consistent conditions
- Additional healthcare datasets for external validation
- Evaluation on quantum hardware, where feasible
The project is for research and educational purposes only. It must not be used for medical diagnosis or treatment decisions.
## 8. Troubleshooting
### 8.1 Module Not Found
Example:
ModuleNotFoundError: No module named 'qiskit'
Suggested fixes:
- Activate the correct virtual environment.
- Install dependencies:
  python -m pip install -r requirements.txt
- Confirm that the script uses the same Python interpreter where the packages were installed.
### 8.2 Qiskit Import or Version Errors
Check installed versions:
python -m pip show qiskit qiskit-machine-learning
Install compatible versions as specified by the project dependency file. Avoid mixing packages from different Python environments.
8.3 Dataset File Not Found
- Confirm that the dataset has been downloaded.
- Check the dataset path in the script.
- Verify the CSV filename and target column name.
### 8.4 Memory or Runtime Issues
- Start with fewer feature configurations when debugging.
- Use a smaller subset for a quick test.
- Reduce optimizer iterations for debugging only, and document configuration changes.
- Close unnecessary applications if memory is limited.
### 8.5 W&B Authentication Error
python -m pip install wandb
wandb login
Do not paste API keys into source files or public repositories.
### 8.6 Results or Plots Not Generated
- Check terminal output for exceptions.
- Confirm the output directory exists or can be created.
- Verify that the experiment completed successfully.
- Check the configured paths for CSV files and plots.
## 9. Acknowledgements
Tools and Technologies Used
- Python
- Qiskit
- Qiskit Machine Learning
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Seaborn
- Weights & Biases (optional)
- Pima Indians Diabetes Dataset
## 10. AI Assistance Disclaimer
ChatGPT was used as an AI assistance tool during project development for:
- Explaining QML and classical machine learning concepts
- Debugging Python and Qiskit implementation issues
- Reviewing experiment design and evaluation metrics
- Improving project documentation and report structure
