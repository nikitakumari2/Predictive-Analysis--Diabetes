# %% [markdown]
# # Part 1: Setup and Data Preparation

# %%
# --- Library Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, make_scorer, recall_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# %%
# --- Data Loading and Initial Inspection ---
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

print("--- Initial Data Info ---")
print(df.info())
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- First 5 Rows ---")
print(df.head())

# %%
# --- Target Variable Preprocessing and Splitting ---
print("\n--- Class Distribution ---")
print(df['Diabetes_012'].value_counts())

df.rename(columns={'Diabetes_012': 'class'}, inplace=True)

X = df.drop('class', axis=1)
y = df['class']

# Stratified split for train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Stratified split for train/validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")

# %% [markdown]
# # Part 2: Exploratory Data Analysis (EDA)

# %%
# --- Histograms for Key Features ---
for col in ['BMI', 'GenHlth', 'Age']:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=col, hue='class', kde=True, multiple="stack", palette="viridis")
    plt.title(f'Distribution of {col} by Class')
    plt.show()

# %%
# --- Box Plots for Key Features ---
for col in ['BMI', 'GenHlth', 'Age']:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='class', y=col, data=df, palette="viridis")
    plt.title(f'Boxplot of {col} by Class')
    plt.show()

# %%
# --- Correlation Matrix ---
plt.figure(figsize=(18, 15))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
plt.title('Correlation Matrix of Features')
plt.show()

# %%
# --- Feature Importance Assessment ---
# Chi-Square Scores
chi2_selector = SelectKBest(chi2, k='all')
chi2_selector.fit(X_train, y_train)
chi2_scores = pd.Series(chi2_selector.scores_, index=X_train.columns)
print("\n--- Chi-Square Scores ---")
print(chi2_scores.sort_values(ascending=False))

# F-value Scores
fvalue_selector = SelectKBest(f_classif, k='all')
fvalue_selector.fit(X_train, y_train)
fvalue_scores = pd.Series(fvalue_selector.scores_, index=X_train.columns)
print("\n--- F-value Scores ---")
print(fvalue_scores.sort_values(ascending=False))

# %% [markdown]
# # Part 3: Sklearn Modeling Pipelines

# %%
# --- Define Preprocessing, Feature Selection, and SMOTE ---
numeric_features = X.columns.tolist()
numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)])

smote = SMOTE(random_state=42)
feature_selector = SelectKBest(chi2)

# --- Define Model Pipelines ---
# Logistic Regression Pipeline
pipeline_lr = ImbPipeline(steps=[
    ('preprocess', preprocessor),
    ('feature_selection', SelectKBest(chi2)),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'))
])

# Random Forest Pipeline
pipeline_rf = ImbPipeline(steps=[
    ('preprocess', preprocessor),
    ('feature_selection', SelectKBest(chi2)),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# XGBoost Pipeline
pipeline_xgb = ImbPipeline(steps=[
    ('preprocess', preprocessor),
    ('feature_selection', SelectKBest(chi2)),
    ('smote', SMOTE(random_state=42)),
    ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
])


# %% [markdown]
# # Part 4: Hyperparameter Tuning

# %%
# --- Tuning Function ---
def tune_model(pipeline, param_grid, X_train, y_train, n_iter=10):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recall_scorer = make_scorer(recall_score, average='macro')

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        cv=kfold,
        scoring=recall_scorer,
        verbose=1,
        n_jobs=-1,
        n_iter=n_iter,
        random_state=42,
        error_score=0
    )
    search.fit(X_train, y_train)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation score (macro recall): {search.best_score_:.4f}")
    return search.best_estimator_

# --- Hyperparameter Grids ---
param_grid_lr = {
    'feature_selection__k': [7, 10, 15, 'all'],
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2']
}

param_grid_rf = {
    'feature_selection__k': [7, 10, 15, 'all'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, 20, None],
    'classifier__min_samples_split': [2, 5, 10]
}

param_grid_xgb = {
    'feature_selection__k': [7, 10, 15, 'all'],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 4, 5],
    'classifier__learning_rate': [0.01, 0.1, 0.2]
}

# %%
# --- Tune Models (Using smaller n_iter for speed) ---
print("--- Tuning Logistic Regression ---")
best_lr = tune_model(pipeline_lr, param_grid_lr, X_train, y_train, n_iter=4)

print("\n--- Tuning Random Forest ---")
best_rf = tune_model(pipeline_rf, param_grid_rf, X_train, y_train, n_iter=4)

print("\n--- Tuning XGBoost ---")
best_xgb = tune_model(pipeline_xgb, param_grid_xgb, X_train, y_train, n_iter=4)

# %% [markdown]
# # Part 5: PyTorch MLP Implementation

# %%
# --- PyTorch Dataset Class ---
class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Preprocess data and apply SMOTE for PyTorch ---
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

train_dataset = DiabetesDataset(X_train_resampled, y_train_resampled)
val_dataset = DiabetesDataset(X_val_processed, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# %%
# --- MLP Model Architecture ---
class MLP(nn.Module):
    def __init__(self, input_size, num_classes=3):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# --- Training Loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(X_train_resampled.shape[1]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

epochs = 10
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # --- Validation Loop ---
    model.eval()
    with torch.no_grad():
        val_losses = []
        all_preds, all_labels, all_probs = [], [], []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            val_losses.append(val_loss.item())

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item():.4f}, Validation Loss: {np.mean(val_losses):.4f}')

# %%
# --- Evaluate PyTorch MLP ---
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

print("\n--- PyTorch MLP Validation Report ---")
print(classification_report(all_labels, all_preds))
print(f"ROC AUC (One-vs-Rest): {roc_auc_score(all_labels, all_probs, multi_class='ovr'):.4f}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1', 'Predicted 2'],
            yticklabels=['Actual 0', 'Actual 1', 'Actual 2'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - PyTorch MLP (Validation)')
plt.show()

# %% [markdown]
# # Part 6: Final Model Evaluation on Test Set

# %%
# --- Evaluation Function ---
def evaluate_pipeline(pipeline, X_test, y_test, title):
    print(f"\n--- {title} ---")
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)

    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Prediabetes', 'Diabetes']))
    try:
        roc_auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
        print(f"ROC AUC (One-vs-Rest): {roc_auc:.4f}")
    except ValueError:
        print("ROC AUC not calculated.")

# %%
# --- Evaluate Best Models on Test Data ---
print("\n--- FINAL TEST SET EVALUATION ---")
evaluate_pipeline(best_lr, X_test, y_test, "Best Logistic Regression on Test Set")
evaluate_pipeline(best_rf, X_test, y_test, "Best Random Forest on Test Set")
evaluate_pipeline(best_xgb, X_test, y_test, "Best XGBoost on Test Set")
# %%
