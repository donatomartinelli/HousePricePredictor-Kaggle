# Kaggle House Prices Prediction Project
## Overview
This repository contains code and documentation for a Kaggle competition focused on predicting house prices. The goal of the competition is to develop a machine learning model that accurately predicts the sale prices of houses based on various features.

## Project Structure
The project is organized into the following directories:

data: Contains the raw and processed datasets.
notebooks: Jupyter notebooks for data exploration, cleaning, and model development.
src: Python scripts for data preprocessing, feature engineering, and model training.
models: Saved models and model evaluation metrics.

##Data Cleaning and Preprocessing
Data Cleaning: The raw datasets are cleaned to handle missing values, outliers, and other anomalies that may affect model performance.

Ordinal Encoding: Categorical variables with an inherent order are encoded using ordinal encoding to preserve the ordinal relationship.

Frequency Encoding: Categorical variables with high cardinality are encoded based on their frequency to reduce dimensionality.

One-Hot Encoding: Remaining categorical variables are one-hot encoded to represent them as binary vectors.

## Feature Engineering
Correlation Analysis: Identifying and removing highly correlated features to improve model interpretability and reduce overfitting.

Mutual Information: Evaluating the mutual information between features and the target variable to select the most informative features for modeling.

## Model Development
Random Forest: Applying the Random Forest algorithm for regression, with hyperparameter tuning to optimize model performance.

Gradient Boosting: Implementing Gradient Boosting for regression, also with hyperparameter tuning for optimal results.

## Results
The final models are evaluated based on relevant metrics such as mean absolute error, mean squared error, and R-squared. Results and insights are documented in the notebooks.
