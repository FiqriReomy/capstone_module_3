# CAPSTONE MODULE 3: E-COMMERCE CUSTOMER CHURN - Analysis and Prediction

## Project Introduction

### Project Overview

This repository contains the capstone project for Module 3, focusing on customer churn analysis and prediction for PT. E-Shop, an e-commerce company based in the UK. The project aims to build a predictive model to identify customers at risk of churning, understand the factors influencing churn, and provide actionable insights to reduce customer attrition.

### Project Structure

The project is organized into several sections, each addressing different aspects of the analysis and model building process:

1. **Business Problem**

   - [Business Background](#business-background)
   - [Problem Statement](#problem-statement)
   - [Goals](#goals)
   - [Analytical Approach](#analytical-approach)

2. **Data Understanding**

   - [Data Information](#data-information)
   - [Descriptive Analysis](#descriptive-analysis)
     - [Numerical Features](#numerical-features)
     - [Categorical Features](#categorical-features)
   - [Missing Value & Miss-Type](#missing-value--miss-type)
   - [Data Distribution](#data-distribution)
     - [Numerical Features](#numerical-features-1)
     - [Categorical Features](#categorical-features-1)
     - [Variable Target (Churn)](#variable-target-churn)
   - [Data Correlation](#data-correlation)
     - [Numerical Features](#numerical-features-2)
     - [Categorical Features](#categorical-features-2)

3. **Data Analysis**

   - [How Does Numerical Feature Affect Customer Churn?](#how-does-numerical-feature-affect-customer-churn)
   - [How Does Categorical Feature Affect Customer Churn?](#how-does-categorical-feature-affect-customer-churn)

4. **Data Pre-processing**

   - [Data Cleaning](#data-cleaning)
     - [Missing Values](#missing-values)
     - [Drop Duplicated Values](#drop-duplicated-values)
     - [Outliers](#outliers)
     - [Save Clean Data](#save-clean-data)
   - [Features Selection](#features-selection)
   - [Features Engineering](#features-engineering)
     - [Encoding](#encoding)
     - [Imputation and Scaling](#imputation-and-scaling)

5. **Model Benchmarking**

   - [Model Selection](#model-selection)
   - [Model Benchmark Evaluation](#model-benchmark-evaluation)
     - [Model Benchmark on Training](#model-benchmark-on-training)
     - [Model Benchmark on Test](#model-benchmark-on-test)
   - [Models Benchmark Evaluation](#models-benchmark-evaluation)

6. **Model Resampling and Hyperparameter Tuning**

   - [Finding Best Resampling Method](#finding-best-resampling-method)
     - [XGBoost Classifier Best Resampling](#xgboost-classifier-best-resampling)
     - [AdaBoost Classifier Best Resampling](#adaboost-classifier-best-resampling)
     - [LGBM Classifier Best Resampling](#lgbm-classifier-best-resampling)
   - [Model Hyperparameter Tuning](#model-hyperparameter-tuning)
     - [Tuning XGBoost Classifier](#tuning-xgboost-classifier)
     - [Tuning AdaBoost Classifier](#tuning-adaboost-classifier)
     - [Tuning LGBM Classifier](#tuning-lgbm-classifier)
   - [Best Model Selection](#best-model-selection)

7. **Best Model Performance Evaluation**

   - [Best Model Score Evaluation](#best-model-score-evaluation)
   - [Saving Best Model](#saving-best-model)
   - [Model Limitation](#model-limitation)
   - [Confusion Metrics Before and After](#confusion-metrics-before-and-after)
   - [Explainable AI For Feature Importance Analysis](#explainable-ai-for-feature-importance-analysis)
   - [Model Performance Testing](#model-performance-testing)

8. **Conclusion and Recommendation**
   - [Conclusion](#conclusion)
   - [Recommendation](#recommendation)

### Business Problem

#### Business Background

PT. E-Shop is a UK-based technology company specializing in e-commerce. Despite significant growth in customer base and transaction volume, the company faces challenges in customer retention amidst increasing competition. Retaining loyal customers is crucial for profitability as returning customers tend to spend more.

#### Problem Statement

Customer acquisition is more costly than retention. A 5% increase in retention can lead to a 25% increase in profitability. Therefore, PT. E-Shop must identify customers likely to churn and take preventive actions.

#### Goals

- Build a model to predict customer churn.
- Minimize retention costs for customers likely to churn.
- Understand characteristics of customers likely to churn.
- Provide targeted treatments to prevent churn.

#### Analytical Approach

The analysis employs machine learning to build a classification model predicting churn probability. Emphasis is placed on minimizing False Negatives to avoid losing customers.

### Data Understanding

#### Data Information

The dataset includes 3264 entries and 11 columns, featuring customer details and churn status.

#### Descriptive Analysis

Data exploration covers numerical and categorical features, missing values, and data distribution.

### Data Pre-processing

#### Data Cleaning

Includes handling missing values, duplicates, and outliers.

#### Features Selection and Engineering

Involves encoding, imputation, and scaling of features.

### Model Benchmarking and Tuning

#### Model Selection and Evaluation

Involves selecting models and evaluating their performance on training and test data.

#### Resampling and Hyperparameter Tuning

Focuses on finding the best resampling methods and tuning model hyperparameters for optimal performance.

### Best Model Performance Evaluation

Includes assessing the best model's score, saving it, and analyzing feature importance.

### Conclusion and Recommendation

Provides final insights and actionable recommendations to reduce customer churn.

### Metric Selection

#### Confusion Matrix

- **True Positive (TP)**: Correctly predicted churn
- **True Negative (TN)**: Correctly predicted not churn
- **False Negative (FN)**: Incorrectly predicted not churn
- **False Positive (FP)**: Incorrectly predicted churn

#### Prediction Error

| **Error Type**                    | **Meaning**                                              | **Outcome**                                               |
| --------------------------------- | -------------------------------------------------------- | --------------------------------------------------------- |
| **False Positive / Type 1 Error** | Model incorrectly predicts churn when actually not churn | Retention cost wasted on loyal customers                  |
| **False Negative / Type 2 Error** | Model incorrectly predicts not churn when actually churn | Loss of recurring revenue and increased acquisition costs |

#### Metric Emphasis

The primary focus is on recall to minimize False Negatives, using the F2 Score to ensure accurate identification of customers likely to churn.

### Repository Contents

- `data/`: Contains the dataset used for analysis.
- `notebooks/`: Jupyter notebooks for data analysis and model building.
- `models/`: Saved models and performance evaluation results.
- `scripts/`: Python scripts for data preprocessing, model training, and evaluation.
- `README.md`: Project introduction and structure.

### Author

- **PURWADHIKA DIGITAL TALENT INCUBATOR**
- **Ahmad Fiqri Oemry**
- **DTI DS 0106**

For detailed analysis and code implementation, refer to the respective sections in this repository.
