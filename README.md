# Kaggle House Prices Prediction Project

## Overview

This repository contains code and documentation for a Kaggle competition focused on predicting house prices. The goal of the competition is to develop a machine learning model that accurately predicts the sale prices of houses based on various features.

## Data Cleaning and Preprocessing

Data Cleaning: The raw datasets are cleaned to handle missing values, outliers, and other anomalies that may affect model performance.

**Ordinal Encoding**: Categorical variables with an inherent order are encoded using ordinal encoding to preserve the ordinal relationship.

**Frequency Encoding**: Categorical variables with high cardinality are encoded based on their frequency to reduce dimensionality.

**One-Hot Encoding**: Remaining categorical variables are one-hot encoded to represent them as binary vectors.

## Feature Engineering

**Correlation Analysis**: Identifying and removing highly correlated features to improve model interpretability and reduce overfitting.

**Mutual Information**: Evaluating the mutual information between features and the target variable to select the most informative features for modeling.

## Model Development

**Random Forest**: Applying the Random Forest algorithm for regression, with hyperparameter tuning to optimize model performance.

**Gradient Boosting**: Implementing Gradient Boosting for regression, also with hyperparameter tuning for optimal results. (Actually the best model with 0.13 RMAE)

## Results
The final results are assessed based on the RMAE metric, and are displayed in a rolling leaderboard on Kaggle. Currently this code puts me in the 68.7th percentile.
Further improvements will come along the way.
