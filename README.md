# Predictive Analysis for Chronic Disease: Diabetes

This repository contains a Jupyter Notebook that explores a dataset related to diabetes and health indicators, performs data preprocessing, EDA, feature selection, and builds several machine learning models for diabetes prediction.

## Overview

The project aims to predict diabetes status (no diabetes, pre-diabetes, or diabetes) based on various health indicators from the BRFSS 2015 dataset.  The notebook covers the following key steps:

1.  **Data Loading and Preparation:** Loads the dataset, handles renaming columns, and splits the data into training, validation, and test sets.  Stratified splitting ensures class balance is maintained across the splits.

2.  **Exploratory Data Analysis (EDA):**  Provides descriptive statistics, class distribution analysis, and ANOVA tests to understand the relationships between features and the target variable. Histograms and boxplots visualize the distribution of key features. A correlation matrix heatmap visualizes feature interdependencies.

3.  **Feature Selection:** Uses Chi-square and ANOVA F-value scores (via `SelectKBest`) to rank features based on their statistical significance in relation to the target variable.

4.  **Model Building and Training:**
    *   Builds pipelines using `sklearn.pipeline.Pipeline` and `imblearn.pipeline.Pipeline` to streamline preprocessing, feature selection, oversampling (using SMOTE to address class imbalance), and model training.
    *   Implements and trains the following models:
        *   Logistic Regression
        *   Random Forest
        *   XGBoost
    *   Uses `RandomizedSearchCV` for hyperparameter tuning of the scikit-learn models, aiming to maximize macro-averaged recall.
    *   Defines a custom `DiabetesDataset` class for PyTorch data loading.

5.  **Model Evaluation:** Defines a function `evaluate_pipeline` to assess model performance on the test set, printing a classification report and ROC AUC score. The best performing tuned scikit learn model (Logistic Regression) is selected and evaluated. A confusion matrix is also created for the PyTorch MLP.

