# AIMLCZCG565_MachineLearningAssignment_2
Machine Learning Assignment -2 Using Model comparison and displays the results in the streamlit.app
# ML Assignment 2: Bank Marketing Classification Using Machine Learning

**1. Problem Statement**

The objective of this project is to develop and evaluate multiple Machine Learning classification models for predicting whether a bank customer belongs to a specific target class based on customer demographic and marketing campaign information.

The project aims to:

Perform data preprocessing and feature engineering.
Train multiple classification algorithms.
Compare model performance using standard evaluation metrics.
Identify the best-performing model for the chosen dataset.

The following classification models were implemented:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbor (KNN) Classifier
Gaussian Naive Bayes Classifier
Random Forest Classifier (Ensemble)

**2. Dataset Description**
Dataset Name : Bank Marketing and Customer Behavior Dataset
Source Kaggle : path = kagglehub.dataset_download("razanihababdellatif/bank-marketing-and-customer-behavior-dataset")

Dataset Overview
The dataset contains information about bank customers and historical marketing campaigns.
The classification target is: Binary classfication

Dataset Characteristics
    Property	Value Number of Instances	45,211
    Number of Features	16
    Target Variable	Class
    Problem Type	Binary Classification
    Missing Values	Handled During Preprocessing


**3. Github Repository Link**
(https://github.com/MManibharathi/AIMLCZCG565_MachineLearningAssignment_2/tree/main)

**4. Models used & Evaluation Metrics**

| ML Model Name                 |   Accuracy |   AUC Score |   Precision |   Recall |   F1 Score |   MCC Score |
|:------------------------------|-----------:|------------:|------------:|---------:|-----------:|------------:|
| Logistic Regression           |   0.899259 |    0.900599 |    0.916314 | 0.974953 |   0.944724 |    0.408434 |
| Decision Tree Classifier      |   0.865974 |    0.692883 |    0.928617 | 0.918848 |   0.923706 |    0.373277 |
| K-Nearest Neighbor Classifier |   0.892845 |    0.804301 |    0.914168 | 0.969693 |   0.941112 |    0.37278  |
| Gaussian Naive Bayes          |   0.869291 |    0.808243 |    0.929211 | 0.922229 |   0.925707 |    0.382761 |
| Random Forest Classifier      |   0.905452 |    0.919502 |    0.923497 | 0.973575 |   0.947875 |    0.462366 |

**5. Observations**

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed well as a baseline model. It provided stable and interpretable results with moderate computational cost. |
| **Decision Tree** | Captured nonlinear relationships effectively but may be prone to overfitting compared to other models. |
| **KNN** | Produced competitive results but required more computation during prediction due to distance calculations. |
| **Naive Bayes** | Fastest model to train and predict. Performance depended on how closely the dataset followed Naive Bayes assumptions. |
| **Random Forest (Ensemble)**| Achieved robust performance by combining multiple decision trees and reducing overfitting. Generally produced the best balance of Accuracy, AUC, F1 Score, and MCC.|
| **Overall Winner** | Random Forest Classifier (Ensemble) (Based on highest Accuracy/AUC/F1/MCC from experimental results). Reason

The Random Forest Classifier typically performs better because:

Combines multiple decision trees.
Reduces overfitting.
Handles nonlinear relationships effectively.
Provides strong generalization performance on unseen data.
Produces consistently high Accuracy, Precision, Recall, F1 Score, and MCC scores. |
