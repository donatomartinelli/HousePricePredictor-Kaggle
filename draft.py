import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

random_seed = 0
np.random.seed(random_seed)

import warnings
warnings.filterwarnings('ignore')


def dataset_summary(datasets, display_columns, display_dtype, display_statistics):
    """
    This function summarizes a dataset by providing essential information such as dataset shape,
    total cells, missing data statistics, columns with missing values, and data type counts.
    
    Parameters:
    datasets (DataFrame): The dataset to be summarized.
    display_columns (str): A flag indicating whether to display dataset columns or not ("y" for yes, "n" for no).
    display_dtype (str): A flag indicating whether to display dtype for each column or not ("y" for yes, "n" for no).
    display_statistics (str): A flag indicating whether to display summary statitsics or not ("y" for yes, "n" for no).
    """

    # Display the entire dataset
    print("Dataset:")
    print(datasets)

    # Get dataset columns and shape
    dataset_columns = list(datasets.columns)
    dataset_shape = datasets.shape

    # Calculate missing data statistics
    missing_data_per_column = datasets.isnull().sum()
    total_cells = np.product(dataset_shape)
    total_missing = missing_data_per_column.sum()
    percent_missing = (total_missing / total_cells) * 100

    # Display dataset shape and missing data statistics
    print("------------------------------------")
    print(f"Dataset Shape: {dataset_shape}")
    print(f"Total Cells: {total_cells}")
    print(f"Total Missing: {total_missing}")
    print(f"Percentage of Missing Data: {percent_missing:.2f}%")
    print("------------------------------------")

    # Display columns with missing values and their counts
    print("Columns with Missing Values:")
    for column in dataset_columns:
        if datasets[column].isnull().sum() > 0:
            print(f" {column}, Missing Values: {datasets[column].isnull().sum()}")

    # Count the occurrence of each data type
    dataset_datatypes = {}
    for column in dataset_columns:
        data_type = datasets[column].dtype
        if data_type in dataset_datatypes:
            dataset_datatypes[data_type] += 1
        else:
            dataset_datatypes[data_type] = 1

    # Display dataset datatypes with their counts
    print("------------------------------------")
    print("Dataset Datatypes with Counts:")
    for data_type, count in dataset_datatypes.items():
        print(f" {data_type}: {count}")
    print("------------------------------------")

    if display_columns == "y":
        # Display dataset columns
        print("Dataset Columns:")
        for column in dataset_columns:
            print(column)
    if display_dtype == "y":
        # Display dataset columns
        print("Dataset Columns Data Types:")
        print(datasets.info())
    if display_statistics == "y":
        # Display dataset columns
        print("Dataset Summary Statistics:")
        print(train.describe().T)

    return ""

train_filepath = "C:/Users/marti/Desktop/projects/HousePricePredictor-Kaggle/train.csv"
test_filepath = "C:/Users/marti/Desktop/projects/HousePricePredictor-Kaggle/test.csv"
train= pd.read_csv(train_filepath)
test= pd.read_csv(test_filepath)

print(dataset_summary(train, display_columns="n", display_dtype="y", display_statistics="y"))
























