# Credit Card Fraud Detection using Machine Learning

## Overview

This project implements an end to end machine learning pipeline for detecting fraudulent credit card transactions using the popular Credit Card Fraud Detection dataset.

The project covers:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Handling class imbalance using SMOTE
* Fraud classification using XGBoost
* Baseline anomaly detection using Isolation Forest
* Model evaluation with ROC-AUC and PR-AUC
* Explainable AI using SHAP

---

## Dataset

Dataset used:

Credit Card Fraud Detection Dataset

The dataset contains anonymized transaction features generated using PCA transformation.

### Features

* `V1` to `V28`

  * PCA transformed numerical features
* `Amount`

  * Transaction amount
* `Time`

  * Seconds elapsed between transactions
* `Class`

  * Target variable
  * `0 = Legitimate`
  * `1 = Fraud`

### Class Imbalance

Fraudulent transactions represent less than 0.2% of the dataset, making this a highly imbalanced classification problem.

---

## Project Structure

```text
project/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   ├── 01_eda.py
│   ├── 02_preprocessing.py
│   ├── 03_modeling.py
│   └── 04_explainability.py
│
├── outputs/
│   ├── plots/
│   ├── processed_data.pkl
│   ├── xgb_model.pkl
│   └── evaluation_metrics.txt
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* Seaborn
* Imbalanced-learn

---

## Workflow

### 1. Exploratory Data Analysis

Performed:

* Class distribution analysis
* Transaction amount analysis
* Time distribution analysis
* Correlation analysis
* Visualization of fraud patterns

Outputs:

* Class distribution plots
* Amount histograms
* Correlation charts

---

### 2. Data Preprocessing

Steps:

* Feature scaling
* Train test split
* SMOTE oversampling on training data
* Saving processed datasets

Important:
SMOTE is applied only on the training set to avoid data leakage.

---

### 3. Modeling

Models implemented:

#### XGBoost Classifier

Primary supervised learning model used for fraud detection.

#### Isolation Forest

Baseline unsupervised anomaly detection model.

Evaluation metrics:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion Matrix

---

### 4. Explainability

SHAP was used to:

* Explain global feature importance
* Visualize feature impact
* Explain individual fraud predictions

Generated outputs:

* SHAP summary plot
* SHAP bar chart
* SHAP waterfall plot

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run notebooks/scripts in order:

### Step 1: EDA

```bash
python notebooks/01_eda.py
```

### Step 2: Preprocessing

```bash
python notebooks/02_preprocessing.py
```

### Step 3: Modeling

```bash
python notebooks/03_modeling.py
```

### Step 4: Explainability

```bash
python notebooks/04_explainability.py
```

---

## Results

The XGBoost model achieved strong fraud detection performance on an extremely imbalanced dataset.

Key observations:

* High ROC-AUC score
* Strong precision recall performance
* SHAP explanations identified the most influential features contributing to fraud predictions

---

## Future Improvements

Possible enhancements:

* Hyperparameter tuning
* Cross validation
* Threshold optimization
* Additional ensemble models
* Real time fraud detection pipeline
* Deployment using Flask or FastAPI

---

## License

This project is intended for educational and research purposes.

---

## Author

Bachelor Thesis Project

Information Engineering

HAW Hamburg
