# Comparative Performance of Random Forest vs. Gradient Boosting for Fraud Detection

Author: **Zeba Samiya**  
Date: **Spring 2025**

## Project Overview

This project investigates and compares two ensemble machine learning techniques—Random Forest (RF) and Gradient Boosting (GB)—for detecting fraudulent transactions in highly imbalanced financial datasets.

## Dataset

- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions over two days in September 2013
- **Fraud Cases**: 492 (~0.17%)
- **Features**:
  - 28 PCA-transformed components (V1–V28)
  - `Time`, `Amount`, and `Class` (1 = Fraud, 0 = Legit)

## Objectives

- Compare **performance** of RF and GB (XGBoost, LightGBM)
- Evaluate **computational efficiency** (training/inference time, memory usage)
- Analyze how each algorithm handles **imbalanced data** using SMOTE
- Perform **feature importance analysis** (feature_importances_ & SHAP)

## Tools & Environment

- **Language**: Python 3.9+
- **Libraries**: `scikit-learn`, `XGBoost`, `LightGBM`, `pandas`, `numpy`, `seaborn`, `matplotlib`, `SHAP`, `imbalanced-learn`
- **Environment**: Jupyter Notebook on GPU 
- **Monitoring Tools**: MLflow, TensorBoard, `psutil`, `df -h`

## Evaluation Metrics

- Precision, Recall, F1-Score
- AUC-ROC Score
- Confusion Matrix
- Training & Inference Time
- Memory & Disk Utilization

## Methodology

- Data preprocessing: scaling, feature selection, train/test split
- Use of SMOTE to address data imbalance
- Hyperparameter tuning via GridSearchCV
- Comparative evaluation of:
  - `RandomForestClassifier`
  - `XGBoostClassifier`
  - `LGBMClassifier`

##  Feature Importance

- **Random Forest**: `.feature_importances_`
- **Gradient Boosting**: SHAP values for detailed interpretability

##  Expected Outcomes

- Gradient Boosting expected to have better precision/recall
- Random Forest expected to train faster and use less memory
- Final discussion on the trade-offs between accuracy and efficiency


> This project is a part of academic research on imbalanced classification and aims to contribute practical insights for real-time fraud detection systems.
