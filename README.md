# Diabetes Prediction from Health Indicators

This project uses the BRFSS 2015 dataset to build and evaluate various machine learning models for predicting a patient's diabetes status (No Diabetes, Prediabetes, or Diabetes).

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [How to Run](#how-to-run)

## Project Overview

The goal of this project is to apply machine learning techniques to a real-world health dataset to classify individuals into three categories: non-diabetic, pre-diabetic, and diabetic. The project explores the challenges of working with imbalanced data and compares the performance of classical machine learning models against a deep learning approach.

## Dataset

The data comes from the **Behavioral Risk Factor Surveillance System (BRFSS) 2015** survey.

- [cite_start]**Size**: 253,680 records and 22 columns.
- [cite_start]**Features**: Key features include `GenHlth`, `HighBP`, `BMI`, `Age`, and `Income`[cite: 20].
- **Target Variable**: `Diabetes_012`, with three classes:
    - [cite_start]`0.0`: No Diabetes (213,703 samples) 
    - [cite_start]`1.0`: Prediabetes (4,631 samples) 
    - [cite_start]`2.0`: Diabetes (35,346 samples) 
- **Challenge**: The dataset is highly imbalanced, with the "Prediabetes" class being a severe minority.

## Methodology

The project follows a standard machine learning workflow:

1.  **Data Preprocessing**: The data was cleaned and split into training (64%), validation (16%), and test (20%) sets. [cite_start]Stratified splitting was used to maintain class proportions[cite: 24].
2.  [cite_start]**Exploratory Data Analysis (EDA)**: Visualizations and statistical tests were used to identify key predictive features and understand their relationship with the target variable[cite: 25, 27, 30]. [cite_start]`GenHlth`, `HighBP`, and `BMI` were found to be highly correlated with diabetes status.
3.  **Modeling**:
    - [cite_start]**Pipelines**: Scikit-learn pipelines were constructed to chain preprocessing (`MinMaxScaler`), feature selection (`SelectKBest`), and oversampling (`SMOTE`) steps with the classifier[cite: 34, 35].
    - **Models Evaluated**:
        - [cite_start]Logistic Regression [cite: 35]
        - [cite_start]Random Forest [cite: 35]
        - [cite_start]XGBoost [cite: 35]
        - [cite_start]A custom Multi-Layer Perceptron (MLP) using PyTorch [cite: 41]
4.  [cite_start]**Hyperparameter Tuning**: `RandomizedSearchCV` was used with 5-fold stratified cross-validation to find the best parameters for each model, optimizing for **macro average recall** to handle the class imbalance[cite: 35, 37].

## Results

The models were evaluated on the held-out test set. The Random Forest model performed best in terms of overall accuracy, while Logistic Regression showed the best-balanced recall across classes.

| Model | Accuracy | ROC AUC | F1-Score (Weighted) | F1-Score (Macro) | Prediabetes Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 66% | 0.768 | 0.73 | 0.43 | **28%** |
| **Random Forest** | **73%** | **0.769** | **0.76** | **0.44** | 11% |
| **XGBoost** | 70% | 0.755 | 0.74 | 0.43 | 13% |

**Key Takeaway**: While all models achieve fair predictive power, they consistently fail to identify the "Prediabetes" class effectively. This highlights the primary challenge of this dataset and suggests that more advanced imbalanced learning techniques or feature engineering may be necessary to improve performance for this critical minority class.

## How to Run

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-link>
    cd <your-repo-folder>
    ```
2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.\.venv\Scripts\Activate`
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the script**:
    Execute the Python script in an interactive environment like VS Code or a Jupyter Notebook to see the outputs and visualizations.
    ```bash
    # Run predict.py in your preferred IDE
    ```
