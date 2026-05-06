"""
Customer Churn Prediction using Machine Learning
Complete pipeline script - runs end to end without errors
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# Change to the script's directory so file paths work correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("1. Importing dependencies - DONE")
print("=" * 70)

# ============================================================
# 2. Data Loading and Understanding
# ============================================================
print("\n" + "=" * 70)
print("2. Data Loading and Understanding")
print("=" * 70)

# Load the dataset - the file is an Excel file (.xlsx)
df = pd.read_excel("WA_Fn-UseC_-Telco-Customer-Churn-Updated.xlsx")

print(f"Dataset shape: {df.shape}")
print("\nFirst 2 rows:")
print(df.head(2))

print("\nDataset info:")
print(df.info())

# Drop customerID column as it's not required for modelling
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

print("\nColumns after dropping customerID:")
print(df.columns.tolist())

# Print unique values for categorical columns
numerical_features_list = ["tenure", "MonthlyCharges", "TotalCharges"]
print("\nUnique values in categorical columns:")
for col in df.columns:
    if col not in numerical_features_list:
        print(col, df[col].unique())
        print("-" * 50)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Handle TotalCharges - it may be object dtype with whitespace values
if df["TotalCharges"].dtype == object:
    print("\nTotalCharges has space values, replacing with 0.0")
    print(f"Number of space values: {len(df[df['TotalCharges'] == ' '])}")
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
    df["TotalCharges"] = df["TotalCharges"].astype(float)
elif df["TotalCharges"].dtype != float:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0.0)

print("\nDataset info after fixing TotalCharges:")
print(df.info())

# Check Churn column value counts
print("\nChurn value counts:")
print(df["Churn"].value_counts())

# ============================================================
# 3. Exploratory Data Analysis (EDA)
# ============================================================
print("\n" + "=" * 70)
print("3. Exploratory Data Analysis (EDA)")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

print("\nDescriptive statistics:")
print(df.describe())

# Histogram plots
def plot_histogram(df, column_name, save_path=None):
    plt.figure(figsize=(5, 3))
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")
    col_mean = df[column_name].mean()
    col_median = df[column_name].median()
    plt.axvline(col_mean, color="red", linestyle="--", label="Mean")
    plt.axvline(col_median, color="green", linestyle="-", label="Median")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    plot_histogram(df, col, f"histogram_{col}.png")
    print(f"  Saved histogram for {col}")

# Box plots
def plot_boxplot(df, column_name, save_path=None):
    plt.figure(figsize=(5, 3))
    sns.boxplot(y=df[column_name])
    plt.title(f"Box Plot of {column_name}")
    plt.ylabel(column_name)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    plot_boxplot(df, col, f"boxplot_{col}.png")
    print(f"  Saved boxplot for {col}")

# Correlation heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png", bbox_inches='tight')
plt.close()
print("  Saved correlation heatmap")

# Countplot for categorical columns
categorical_columns = df.select_dtypes(include="object").columns
for col in categorical_columns:
    plt.figure(figsize=(5, 3))
    sns.countplot(x=df[col])
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"countplot_{col}.png", bbox_inches='tight')
    plt.close()
    print(f"  Saved countplot for {col}")

# ============================================================
# 4. Data Preprocessing
# ============================================================
print("\n" + "=" * 70)
print("4. Data Preprocessing")
print("=" * 70)

# Label encoding of target column
df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
print(f"\nChurn value counts after encoding:")
print(df["Churn"].value_counts())

# Label encoding of categorical features
object_columns = df.select_dtypes(include="object").columns
print(f"\nObject columns to encode: {object_columns.tolist()}")

encoders = {}
for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

# Save the encoders to a pickle file
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print("  Saved encoders to encoders.pkl")

print("\nEncoded dataset (first 5 rows):")
print(df.head())

# ============================================================
# 5. Training and Test Data Split + SMOTE
# ============================================================
print("\n" + "=" * 70)
print("5. Training and Test Data Split + SMOTE")
print("=" * 70)

# Split features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {y_train.shape[0]}")
print(f"Test set size: {y_test.shape[0]}")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Training set size: {y_train_smote.shape[0]}")
print(f"Training set class distribution:")
print(y_train_smote.value_counts())

# ============================================================
# 6. Model Training
# ============================================================
print("\n" + "=" * 70)
print("6. Model Training")
print("=" * 70)

# Dictionary of models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# Perform 5-fold cross validation for each model
cv_scores = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name} with default parameters...")
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
    cv_scores[model_name] = scores
    print(f"  {model_name} cross-validation accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    print(f"  Individual fold scores: {scores}")

# Select best model (Random Forest based on previous results)
print("\n\nBest model: Random Forest")
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote, y_train_smote)
print("  Random Forest trained successfully!")

# ============================================================
# 7. Model Evaluation
# ============================================================
print("\n" + "=" * 70)
print("7. Model Evaluation")
print("=" * 70)

print(f"\nTest set class distribution:")
print(y_test.value_counts())

# Evaluate on test data
y_test_pred = rfc.predict(X_test)

print(f"\nAccuracy Score: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Save the trained model as a pickle file
model_data = {"model": rfc, "features_names": X.columns.tolist()}
with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
print("  Saved model to customer_churn_model.pkl")

# ============================================================
# 8. Predictive System Demo
# ============================================================
print("\n" + "=" * 70)
print("8. Predictive System Demo")
print("=" * 70)

# Load the saved model and the feature names
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]
print(f"\nLoaded model: {loaded_model}")
print(f"Feature names: {feature_names}")

# Sample prediction
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

input_data_df = pd.DataFrame([input_data])

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Encode categorical features using the saved encoders
for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

# Make a prediction
prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

print(f"\nSample Input: {input_data}")
print(f"\nPrediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
print(f"Prediction Probability: {pred_prob}")

print("\n" + "=" * 70)
print("ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 70)
