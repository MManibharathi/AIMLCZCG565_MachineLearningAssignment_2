import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Machine Learning Model Evaluation App",
    layout="wide"
)

st.title("Machine Learning Model Evaluation App")

st.write(
    "Upload test data, select a machine learning model, and view evaluation metrics, "
    "confusion matrix, classification report, and comparison of different models."
)

# ============================================================
# Model Paths Based on Your Folder Structure
# ============================================================

MODEL_DIR = "model"

model_files = {
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
    "Decision Tree Classifier": os.path.join(MODEL_DIR, "decision_tree_classifier.pkl"),
    "K-Nearest Neighbor Classifier": os.path.join(MODEL_DIR, "k_nearest_neighbor_classifier.pkl"),
    "Gaussian Naive Bayes": os.path.join(MODEL_DIR, "gaussian_naive_bayes.pkl"),
    "Random Forest Classifier": os.path.join(MODEL_DIR, "random_forest_classifier.pkl")
}

feature_columns_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

# ============================================================
# Sidebar Controls
# ============================================================

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

selected_model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(model_files.keys())
)

st.write("Selected Model:", selected_model_name)

# ============================================================
# Function to Plot Confusion Matrix
# ============================================================

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()
    return fig

# ============================================================
# Function to Prepare Test Data
# ============================================================

def prepare_test_data(df, target_column):
    y_test = df[target_column]
    X_test = df.drop(columns=[target_column])

    if os.path.exists(feature_columns_path):
        feature_columns = joblib.load(feature_columns_path)

        missing_columns = [col for col in feature_columns if col not in X_test.columns]
        extra_columns = [col for col in X_test.columns if col not in feature_columns]

        if len(missing_columns) > 0:
            st.warning("Some required feature columns are missing in uploaded test data. Missing columns are filled with 0.")
            st.write(missing_columns)

        if len(extra_columns) > 0:
            st.info("Extra columns found in uploaded test data. These columns will be ignored.")
            st.write(extra_columns)

        X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_test = scaler.transform(X_test)

    return X_test, y_test

# ============================================================
# Function to Evaluate Model
# ============================================================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    auc_score = None

    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

            if y_prob.shape[1] == 2:
                auc_score = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc_score = roc_auc_score(y_test, y_prob, multi_class="ovr")

        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            auc_score = roc_auc_score(y_test, y_score)

    except Exception:
        auc_score = None

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    return y_pred, accuracy, auc_score, precision, recall, f1, mcc

# ============================================================
# Main App Logic
# ============================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("CSV uploaded successfully!")

    st.subheader("Uploaded Test Dataset Preview")
    st.write("Dataset Shape:", df.shape)
    st.dataframe(df.head())

    # Better target column default selection
    possible_target_columns = ["target", "label", "class", "y", "deposit", "Output", "output"]

    default_target_index = len(df.columns) - 1

    for col in possible_target_columns:
        if col in df.columns:
            default_target_index = list(df.columns).index(col)
            break

    target_column = st.sidebar.selectbox(
        "Select Target Column",
        df.columns,
        index=default_target_index
    )

    st.write("Selected Target Column:", target_column)

    st.warning(
        "Important: Do not select feature columns like V1, V2, V3 as target. "
        "Select the actual output column, for example y, target, label, class, or deposit."
    )

    try:
        X_test, y_test = prepare_test_data(df, target_column)

        selected_model_file = model_files[selected_model_name]

        if not os.path.exists(selected_model_file):
            st.error(f"Model file not found: {selected_model_file}")
        else:
            model = joblib.load(selected_model_file)
            st.success(f"Model loaded successfully: {selected_model_name}")

            y_pred, accuracy, auc_score, precision, recall, f1, mcc = evaluate_model(
                model,
                X_test,
                y_test
            )

            # ========================================================
            # Evaluation Metrics
            # ========================================================

            st.subheader("Evaluation Metrics")

            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.6f}")
            col2.metric("AUC Score", f"{auc_score:.6f}" if auc_score is not None else "N/A")
            col3.metric("Precision", f"{precision:.6f}")
            col4.metric("Recall", f"{recall:.6f}")
            col5.metric("F1 Score", f"{f1:.6f}")
            col6.metric("MCC Score", f"{mcc:.6f}")

            # ========================================================
            # Confusion Matrix
            # ========================================================

            st.subheader("Confusion Matrix")

            labels = sorted(y_test.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            fig = plot_confusion_matrix(cm, labels)
            st.pyplot(fig)

            # ========================================================
            # Classification Report
            # ========================================================

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
           
    except Exception as e:
        st.error("Error occurred while evaluating the model.")
        st.exception(e)

else:
    st.info("Please upload a test CSV file from the sidebar to start evaluation.")