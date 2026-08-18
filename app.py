import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
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

st.set_page_config(
    page_title="ML Model Assignment-2 Evaluation App",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Model Evaluation App")
st.write("Upload test data, select a model, and view evaluation results.")

# ---------------------------------------------------------
# Model file mapping
# ---------------------------------------------------------
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree Classifier": "decision_tree.pkl",
    "K-Nearest Neighbor Classifier": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Random Forest Classifier": "random_forest.pkl"
}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("App Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

selected_model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(model_files.keys())
)

target_column = st.sidebar.text_input(
    "Enter target column name",
    value="target"
)

# ---------------------------------------------------------
# Main logic
# ---------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Dataset")
    st.write("Dataset shape:", df.shape)
    st.dataframe(df.head())

    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in uploaded CSV.")
        st.write("Available columns are:")
        st.write(df.columns.tolist())
    else:
        X_test = df.drop(columns=[target_column])
        y_test = df[target_column]

        model_path = model_files[selected_model_name]

        try:
            model = joblib.load(model_path)

            st.success(f"Loaded model: {selected_model_name}")

            y_pred = model.predict(X_test)

            # Probability prediction for AUC
            auc_score = None

            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    auc_score = roc_auc_score(y_test, y_prob)
                except Exception:
                    auc_score = None
            else:
                try:
                    y_score = model.decision_function(X_test)
                    auc_score = roc_auc_score(y_test, y_score)
                except Exception:
                    auc_score = None

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)

            st.subheader("Evaluation Metrics")

            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.6f}")
            col2.metric("AUC Score", f"{auc_score:.6f}" if auc_score is not None else "N/A")
            col3.metric("Precision", f"{precision:.6f}")
            col4.metric("Recall", f"{recall:.6f}")
            col5.metric("F1 Score", f"{f1:.6f}")
            col6.metric("MCC Score", f"{mcc:.6f}")

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax
            )
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")
            ax.set_title(f"Confusion Matrix - {selected_model_name}")

            st.pyplot(fig)

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

        except FileNotFoundError:
            st.error(f"Model file not found: {model_path}")
            st.write("Please make sure the model `.pkl` files are available in the same folder as app.py.")

        except Exception as e:
            st.error("An error occurred while evaluating the model.")
            st.exception(e)

else:
    st.info("Please upload a test CSV file from the sidebar.")