import os
import glob
import warnings
import joblib
import kagglehub

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings("ignore")


# ============================================================
# 1. Download Dataset from Kaggle
# ============================================================

path = kagglehub.dataset_download(
    "razanihababdellatif/bank-marketing-and-customer-behavior-dataset"
)

print("Path to dataset files:", path)


# ============================================================
# 2. Automatically Find CSV File
# ============================================================

csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

if len(csv_files) == 0:
    raise FileNotFoundError("No CSV file found in the downloaded dataset folder.")

print("\nCSV files found:")
for file in csv_files:
    print(file)

csv_file = csv_files[0]
print("\nUsing CSV file:", csv_file)


# ============================================================
# 3. Load Dataset
# ============================================================

df = pd.read_csv(csv_file)

print("\nInitial Dataset Shape:", df.shape)
print("\nDataset Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 4. Clean Column Names
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# 5. Identify Target Column
# ============================================================

possible_target_columns = [
    "Class",
    "class",
    "y",
    "deposit",
    "subscribed",
    "response",
    "target"
]

target_col = None

for col in possible_target_columns:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("\nAvailable columns:", df.columns.tolist())
    raise ValueError(
        "Target column not found. Please check the dataset column name manually."
    )

print("\nTarget Column:", target_col)


# ============================================================
# 6. Remove Unnecessary Spaces from Text Columns
# ============================================================

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip()


# ============================================================
# 7. Convert Target to Binary Values
# ============================================================

print("\nTarget value counts before encoding:")
print(df[target_col].value_counts())

if df[target_col].dtype == "object":
    df[target_col] = df[target_col].astype(str).str.lower().str.strip()

    target_mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0
    }

    df[target_col] = df[target_col].map(target_mapping)
else:
    df[target_col] = df[target_col].astype(int)

if df[target_col].isnull().sum() > 0:
    raise ValueError(
        "Target column contains unknown values. Please check target values."
    )

print("\nTarget value counts after encoding:")
print(df[target_col].value_counts())


# ============================================================
# 8. Separate Features and Target
# ============================================================

X = df.drop(columns=[target_col])
y = df[target_col]


# ============================================================
# 9. Handle Missing Values
# ============================================================

X = X.replace("unknown", np.nan)
X = X.replace("Unknown", np.nan)
X = X.replace("UNKNOWN", np.nan)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumerical Columns:", numeric_cols)
print("\nCategorical Columns:", categorical_cols)

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])


# ============================================================
# 10. One-Hot Encode Categorical Features
# ============================================================

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print("\nFinal Feature Shape after Encoding:", X.shape)


# ============================================================
# 11. Check Assignment Requirements
# ============================================================

num_instances = X.shape[0]
num_features = X.shape[1]

print("\nAssignment Requirement Check:")
print("Number of Instances:", num_instances)
print("Number of Features:", num_features)

if num_instances < 500:
    raise ValueError("Dataset does not satisfy minimum instance size of 500.")

if num_features < 12:
    raise ValueError("Dataset does not satisfy minimum feature size of 12.")

print("\nDataset satisfies assignment requirements.")


# ============================================================
# 12. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# ============================================================
# 13. Feature Scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 14. Define ML Classification Models
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbor Classifier": KNeighborsClassifier(n_neighbors=5),
    "Gaussian Naive Bayes": GaussianNB(),
    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ============================================================
# 15. Create Folder to Save Models
# ============================================================

os.makedirs("model", exist_ok=True)

joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")


# ============================================================
# 16. Train Models and Calculate Metrics
# ============================================================

results = []

for model_name, model in models.items():
    print("\nTraining:", model_name)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = roc_auc_score(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy,
        "AUC Score": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC Score": mcc
    })

    model_file_name = model_name.lower().replace(" ", "_").replace("-", "_")
    joblib.dump(model, f"model/{model_file_name}.pkl")

    print(model_name, "saved successfully.")


# ============================================================
# 17. Display Final Results
# ============================================================

results_df = pd.DataFrame(results)

print("\n============================================================")
print("FINAL MODEL EVALUATION RESULTS")
print("============================================================")
print(results_df)

print("\nMarkdown Table for README.md:")
print(results_df.to_markdown(index=False))


# ============================================================
# 18. Save Results and Test Data
# ============================================================

results_df.to_csv("model_evaluation_results.csv", index=False)

test_data = X_test.copy()
test_data["Actual_Target"] = y_test.values
test_data.to_csv("test_data.csv", index=False)

print("\nFiles created successfully:")
print("1. model_evaluation_results.csv")
print("2. test_data.csv")
print("3. model folder with trained models")
print("4. scaler.pkl")
print("5. feature_columns.pkl")
