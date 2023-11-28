import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# Setting display options for pandas
#pd.set_option('display.max_columns', 200)  # Setting the maximum number of displayed columns to 200
pd.set_option('display.max_rows', 200)     # Setting the maximum number of displayed rows to 200

# Importing warnings module and setting to ignore warnings
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

train_filepath = "C:/Users/marti/Desktop/projects/House-Prices--Advanced-Regression-Techniques/train.csv"
test_filepath = "C:/Users/marti/Desktop/projects/House-Prices--Advanced-Regression-Techniques/test.csv"
train= pd.read_csv(train_filepath)
test= pd.read_csv(test_filepath)

print("-"*50)
print("INITIAL STATE OF THE DATASET")
print("-"*50)
print(dataset_summary(train, display_columns="n", display_dtype="y", display_statistics="y"))


# Drop the 'Id' column and modify the DataFrame in place
train.drop(columns=['Id'], inplace=True)

# Drop duplicate rows from the DataFrame in place
train.drop_duplicates(inplace=True)

train['LotFrontage'].fillna(0, inplace=True)
train['MasVnrArea'].fillna(0, inplace=True)
train['GarageYrBlt'].fillna(0, inplace=True)

columns_to_original_encode = ['GarageType', 'LotShape', 'LandContour', 
                              'LandSlope', 'ExterQual', 'ExterCond', 
                              'BsmtQual', 'BsmtCond', 'BsmtExposure',
                              'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
                              'KitchenQual', 'FireplaceQu', 'GarageFinish',
                              'GarageQual', 'GarageCond', 'PoolQC',
                              'Fence']

# Initialize the ordinal encoder
encoder = OrdinalEncoder()
for column in columns_to_original_encode:

    # Handle missing values by replacing with "NA" and then encoding
    train[column].fillna("NA", inplace=True)
    train[column] = encoder.fit_transform(train[[column]])

columns_to_frequency_encode = ['MSZoning', 'Utilities', 'LotConfig',
                               'Neighborhood', 'Condition1', 'Condition2',
                               'BldgType', 'HouseStyle', 'RoofStyle',
                               'RoofMatl', 'Exterior1st', 'Exterior2nd',
                               'MasVnrType', 'Foundation', 'Heating',
                               'Electrical', 'Functional', 'MiscFeature',
                               'SaleType', 'SaleCondition']

for column in columns_to_frequency_encode:
    train[column].fillna("NA", inplace=True)

    # Calculate frequency of each category in the current column
    category_frequencies = train[column].value_counts(normalize=True)

    # Replace categories with their frequencies
    train[column] = train[column].map(category_frequencies)


columns_to_one_hot_encode = ['Street', 'Alley', 'PavedDrive', 'CentralAir']

# Create an empty DataFrame to store the one-hot encoded columns
one_hot_encoded_df = pd.DataFrame()
for column in columns_to_one_hot_encode:
    train[column].fillna("NA", inplace=True)

    # Apply one-hot encoding to the current column
    one_hot_encoded_column = pd.get_dummies(train[column], prefix=column, prefix_sep='_')

    # Concatenate the one-hot encoded column with the new DataFrame
    one_hot_encoded_df = pd.concat([one_hot_encoded_df, one_hot_encoded_column], axis=1)

# Convert boolean values to integers (0s and 1s)
one_hot_encoded_df = one_hot_encoded_df.astype(int)

# Concatenate the one-hot encoded DataFrame with the original DataFrame
train = pd.concat([train, one_hot_encoded_df], axis=1)

# Drop the original columns that were one-hot encoded
train.drop(columns=columns_to_one_hot_encode, inplace=True)

print("-"*50)
print("STATE OF THE DATASET AFTER CATEGORICAL ENCODING AND HANDLING MISSING VALUES")
print("-"*50)
print(dataset_summary(train, display_columns="n", display_dtype="y", display_statistics="y"))

#subset containig all original numerical features plus those who then went through original and freuqency encoding
df_num = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
          'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
          '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
          'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
          'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
          '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold','GarageType', 
          'LotShape', 'LandContour', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
          'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 
          'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC','Fence', 'MSZoning', 'Utilities', 'LotConfig',
          'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
          'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
          'Electrical', 'Functional', 'MiscFeature', 'SaleType', 'SaleCondition', 'SalePrice']
subset_df = train.loc[:,df_num]


# Display descriptive statistics for the 'SalePrice' column
print("Descriptive statistics for the 'SalePrice' column")
print(train['SalePrice'].describe())

# Plot a histogram for the 'SalePrice' column using seaborn
sns.distplot(train['SalePrice'], hist_kws={'alpha': 0.4}, bins=100, color='r')
plt.show()

# Calculate the correlation matrix for numerical features
print("Train Correlation")
print(subset_df.corr())

# Plot a heatmap of the correlation matrix
cormat = subset_df.corr()
paper = plt.figure(figsize=(7, 8))
sns.set(font_scale=1.2)
sns.heatmap(cormat, cmap="coolwarm", cbar=True, vmax=1, square=True)
plt.title("Correlation Heatmap")
plt.show()

# Plot histograms for numerical features
#subset_df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8, color='Red')
#plt.show()

# Select the best numerical features based on correlation with 'SalePrice'
subset_df_corr = subset_df.corr()['SalePrice'][:-1]
best_num_features = subset_df_corr[abs(subset_df_corr) > 0.4].sort_values(ascending=False)
print(f"There are {len(best_num_features)} best features with SalePrice:\n\n{best_num_features}")

# Calculate the correlation matrix for numerical features
correlation_matrix = subset_df.corr()
sales_price_corr = correlation_matrix["SalePrice"].sort_values(ascending=False)
sale_price_corr_df = pd.DataFrame(sales_price_corr)

# Plot a heatmap of correlation with SalePrice
plt.figure(figsize=(10, 8))
sns.heatmap(sale_price_corr_df, annot=True, cmap="coolwarm", cbar=True)
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Correlation Heatmap with SalePrice')
plt.show()

# Initialize a StandardScaler to standardize the numerical features
scaler = StandardScaler()

# Standardize the numerical features in the DataFrame 'df_num'
df_num_scaled = scaler.fit_transform(subset_df)

# Define the target column for which we will calculate mutual information
target_column = 'SalePrice'

# Calculate mutual information between standardized numerical features and the target
mi = mutual_info_regression(df_num_scaled, subset_df[target_column])

# Create a Series to store mutual information with feature names as indices
mi_series = pd.Series(mi, index=subset_df.columns)

# Print mutual information in descending order
print("Mutual Information:")
print(mi_series.sort_values(ascending=False))


features_train_A = ['OverallQual', 'GrLivArea' ,'GarageCars', 'TotalBsmtSF', 'GarageArea', 
                   'YearBuilt', 'Neighborhood','BsmtQual', 'KitchenQual', 'ExterQual' ,
                   '1stFlrSF', 'MSSubClass', 'GarageFinish', 'FullBath', 'GarageYrBlt',
                   'YearRemodAdd', 'TotRmsAbvGrd', 'SalePrice']
train_A = train.loc[:, features_train_A]
print()


"""
At this point, i have a dataframe containing the best features for both mutual information rate and correlation rate
you have to calculate this also for cat variables that were frequency and original encoded, not oe-hot encoded
then modify the best features for both mutual information rate and correlation rate

you have to just structure the code better to make it more readable, 
then understand the remaining lines what asks you to plot

then build 3 neural networks and more models 

one with the best features + onehot encoded variables
one with the best features 

work with each single neural network and hyperparameter tune them

once you're done, compare the results and submit the results of the best one, then prepare a pdf file and post it on your linkedin profile
then send it to the Quant team and asks for suggestions

then congrats, the project is done (you'll come back eventually to optimize it but you can be over with it)

"""




