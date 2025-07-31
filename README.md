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

- **Size**: 253,680 records and 22 columns.
- **Features**: Key features include `GenHlth`, `HighBP`, `BMI`, `Age`, and `Income`.
- **Target Variable**: `Diabetes_012`, with three classes:
    - `0.0`: No Diabetes (213,703 samples)
    - `1.0`: Prediabetes (4,631 samples)
    - `2.0`: Diabetes (35,346 samples)
- **Challenge**: The dataset is highly imbalanced, with the "Prediabetes" class being a severe minority.

## Methodology

The project follows a standard machine learning workflow:

1.  **Data Preprocessing**: The data was cleaned and split into training (64%), validation (16%), and test (20%) sets. Stratified splitting was used to maintain class proportions.
2.  **Exploratory Data Analysis (EDA)**: Visualizations and statistical tests were used to identify key predictive features and understand their relationship with the target variable. `GenHlth`, `HighBP`, and `BMI` were found to be highly correlated with diabetes status.
3.  **Modeling**:
    - **Pipelines**: Scikit-learn pipelines were constructed to chain preprocessing (`MinMaxScaler`), feature selection (`SelectKBest`), and oversampling (`SMOTE`) steps with the classifier.
    - **Models Evaluated**:
        - Logistic Regression
        - Random Forest
        - XGBoost
        - A custom Multi-Layer Perceptron (MLP) using PyTorch
4.  **Hyperparameter Tuning**: `RandomizedSearchCV` was used with 5-fold stratified cross-validation to find the best parameters for each model, optimizing for **macro average recall** to handle the class imbalance.

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
    Execute the Python script (`predict.py`) in an interactive environment like VS Code or a Jupyter Notebook to see the outputs and visualizations.
