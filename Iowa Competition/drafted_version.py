"""
1 - Import library
"""

# Importing necessary libraries
import pandas as pd  # Importing pandas library and aliasing it as pd
import numpy as np   # Importing numpy library and aliasing it as np
import seaborn as sns  # Importing seaborn library and aliasing it as sns
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot and aliasing it as plt

# Importing warnings module and setting to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Importing specific functions/classes from scikit-learn
from sklearn.preprocessing import OrdinalEncoder  

# Importing specific functions/classes from plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# Setting display options for pandas
pd.set_option('display.max_columns', 200)  # Setting the maximum number of displayed columns to 200
pd.set_option('display.max_rows', 200)     # Setting the maximum number of displayed rows to 200

# File paths for the dataset
train_filepath = "C:/Users/donat/Desktop/Programming Projects/Python Projects/Iowa_Competition/train.csv"
test_filepath = "C:/Users/donat/Desktop/Programming Projects/Python Projects/Iowa_Competition/test.csv"

# Reading the datasets using pandas
df = pd.read_csv(train_filepath)  # Reading the training dataset
test = pd.read_csv(test_filepath)  # Reading the test dataset

# Printing the shape of the loaded datasets
print(f"Full DataSet Shape is : {df.shape}")
print(f"Full Test DataSet Shape is : {test.shape}")

"""
2 - Data Preprocessing
Data preprocessing is a critical step in any machine learning project. Here, we will clean the data, handle missing values, and transform variables to ensure they are suitable for model training.
"""

# Drop the 'Id' column and modify the DataFrame in place
df.drop(columns=['Id'], inplace=True)

# Display summary statistics for the DataFrame
print(df.describe().T)

# Drop duplicate rows from the DataFrame in place
df.drop_duplicates(inplace=True)

# Calculate the number of missing values for each column
missing = df.isnull().sum()

# Calculate the length of the DataFrame
lendf = len(df)

# Calculate the percentage of missing values for each column
perc = (missing / lendf) * 100

# Identify columns with more than 40% missing values
col_nam = [i for i, j in perc.items() if j >= 40]

# Print the list of columns with more than 40% missing values
print(f"List of Columns with more than 40% missing values: {col_nam}")

# Drop columns with more than 40% missing values from the DataFrame
df.drop(columns=col_nam, inplace=True)

# Display the shape of the DataFrame after dropping columns
print(f"Shape of DataFrame after dropping columns: {df.shape}")

""" Target Variable Distribution """

# Display descriptive statistics for the 'SalePrice' column
print(df['SalePrice'].describe())

# Plot a histogram for the 'SalePrice' column using seaborn
sns.distplot(df['SalePrice'], hist_kws={'alpha': 0.4}, bins=100, color='r')
plt.show()

# Display unique data types present in the DataFrame
print(df.dtypes.unique())

# Select numerical features (integers and floats)
df_num = df.select_dtypes(include=['int64', 'float64'])
print(df_num.head())  # Display the first few rows of numerical features

# Select categorical features (object data type)
df_cat = df.select_dtypes('O')
print(df_cat.head(2))  # Display the first few rows of categorical features

# Calculate the count of missing values for each column
df_missing = df.isnull().sum()

# Create a backup copy of the DataFrame
backup = df.copy()

# Fill missing values with the mode for each column
filled_df = df.fillna(df.mode().iloc[0])

# Calculate missing values count for the filled DataFrame
filled_missing = filled_df.isnull().sum()

# Display information about numerical features
print(df_num.info())

# Calculate the correlation matrix for numerical features
print(df_num.corr())

# Plot a heatmap of the correlation matrix
cormat = df_num.corr()
paper = plt.figure(figsize=(7, 8))
sns.set(font_scale=1.2)
sns.heatmap(cormat, cmap="coolwarm", cbar=True, linewidths=1, linecolor='black', vmax=1, square=True)
plt.title("Correlation Heatmap")
plt.show()

# Plot histograms for numerical features
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8, color='Red')
plt.show()



# Select the best numerical features based on correlation with 'SalePrice'
df_num_corr = df_num.corr()['SalePrice'][:-1]
best_num_features = df_num_corr[abs(df_num_corr) > 0.4].sort_values(ascending=False)
print(f"There are {len(best_num_features)} best features with SalePrice:\n\n{best_num_features}")

# Calculate the correlation matrix for numerical features
correlation_matrix = df_num.corr()
sales_price_corr = correlation_matrix["SalePrice"].sort_values(ascending=False)
sale_price_corr_df = pd.DataFrame(sales_price_corr)

# Plot a heatmap of correlation with SalePrice
plt.figure(figsize=(10, 8))
sns.heatmap(sale_price_corr_df, annot=True, cmap="coolwarm", cbar=True)
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Correlation Heatmap with SalePrice')
plt.show()

# Print the indices of the best numerical features
print(best_num_features.index)

# Create a new DataFrame with selected best numerical features
best_num_featurelist = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
                        '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
                        'GarageYrBlt', 'MasVnrArea', 'Fireplaces', 'SalePrice']
df_best_num_feature = df.loc[:, best_num_featurelist].copy()
print(df_best_num_feature)

""" Feature to Feature Relationship """

# Plot a heatmap of correlation for the selected numerical features
corr = df_best_num_feature.corr()
paper = plt.figure(figsize=(12, 8))
# Plot heatmap for correlations above 0.5 or below -0.4
sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)],
            cmap='winter', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)
plt.show()

# Drop specific numerical columns from the DataFrame
df_best_num_feature = df_best_num_feature.drop(['GarageYrBlt', 'MasVnrArea', 'Fireplaces'], axis=1)
print(df_best_num_feature.columns)  # Display column names after dropping

# Plot a heatmap of correlation for the modified numerical features
corr = df_best_num_feature.corr()
paper = plt.figure(figsize=(12, 8))
# Plot heatmap for correlations above 0.5 or below -0.4
sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)],
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)
plt.show()

# Plot pairplot for the selected numerical features
sns.pairplot(df_best_num_feature)
plt.show()

# Display count of missing values for each column in df_best_num_feature
print(df_best_num_feature.isnull().sum())

# Display the first few rows of the categorical features DataFrame
df_cat.head()

# Display the number of unique values for each categorical column
df_cat.nunique()

# Display value counts for the 'Neighborhood' column
df_cat['Neighborhood'].value_counts()

# Create a box plot for 'SalePrice' vs 'Neighborhood' using plotly express
fig = px.box(df, x='Neighborhood', y=df['SalePrice'])
# Update layout for the plot
fig.update_layout(
    title='Sale Price Distribution by Neighborhood',
    xaxis=dict(title='Neighborhood'),
    yaxis=dict(title='Sale Price'),
    xaxis_tickangle=-45,
    width=800,
    height=400,
)
fig.show()

# Drop the 'Neighborhood' column from df_cat
df_cat.drop(columns=['Neighborhood'], inplace=True)

# Display the number of unique values for each categorical column after dropping 'Neighborhood'
df_cat.nunique()

# Display value counts for the 'Exterior1st' column
df_cat['Exterior1st'].value_counts()

# Display value counts for the 'Exterior2nd' column
df_cat['Exterior2nd'].value_counts()

# Create a histogram for 'SalePrice' vs 'Exterior2nd' using plotly express
fig = px.histogram(df_cat, x='Exterior2nd', y=df['SalePrice'])
# Update layout for the plot
fig.update_layout(
    title='Sale Price Distribution by Exterior2nd',
    xaxis=dict(title='Exterior2nd'),
    yaxis=dict(title='Sale Price'),
    xaxis_tickangle=-45,
    width=800,
    height=400,
)
fig.show()

# Create a histogram for 'SalePrice' vs 'Exterior1st' using plotly express
fig = px.histogram(df_cat, x='Exterior1st', y=df['SalePrice'])
# Update layout for the plot
fig.update_layout(
    title='Sale Price Distribution by Exterior1st',
    xaxis=dict(title='Exterior1st'),
    yaxis=dict(title='Sale Price'),
    xaxis_tickangle=-45,
    width=800,
    height=400,
)
fig.show()

# Create new binary columns indicating presence of specific materials in 'Exterior1st' and 'Exterior2nd'
# Drop original 'Exterior1st' and 'Exterior2nd' columns
for material in df_cat['Exterior1st'].unique():
    df_cat[f'Has_{material}_Exterior'] = (df_cat['Exterior1st'] == material) | (df_cat['Exterior2nd'] == material)
df_cat.drop(['Exterior1st', 'Exterior2nd'], axis=1, inplace=True)

# Iterate through selected columns, display their value counts
for column in df_cat.columns:
    if df_cat[column].nunique() == 2:
        print(df_cat[column].value_counts())

# Create an empty list to store column names
selected_columns = []

# Loop through the categorical columns and select relevant ones
for column in df_cat.columns:
    if df_cat[column].nunique() == 2:
        value_counts = df_cat[column].value_counts()
        print(f"Column: {column}")
        print(value_counts)
        if value_counts.iloc[1] < 50:
            selected_columns.append(column)

# Print the selected columns
print("Selected Columns:", selected_columns)

# Display value counts for selected categorical columns
for i in df_cat[selected_columns]:
    print(df_cat[i].value_counts())

# Drop irrelevant categorical columns from df_cat
df_cat.drop(columns=(selected_columns), inplace=True)
print(df_cat.shape)  # Display the shape after dropping

# Calculate the number of rows needed for subplotting categorical count plots
num_rows = (len(df_cat.columns) + 2) // 3
fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
plt.subplots_adjust(hspace=0.5)

# Loop through categorical columns and create count plots
for i, column in enumerate(df_cat.columns):
    row_idx = i // 3
    col_idx = i % 3
    ax = axes[row_idx, col_idx]
    
    sns.countplot(data=df_cat, x=column, ax=ax)
    ax.set_title(f'Count Plot for {column}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

# Remove any empty subplots
for i in range(len(df_cat.columns), num_rows * 3):
    fig.delaxes(axes.flatten()[i])

plt.show()

# Display count of missing values for each column in df_cat
print(df_cat.isnull().sum())

# Fill missing values with the mode for each column in df_cat
cat_filled_df = df_cat.fillna(df_cat.mode().iloc[0])
print(cat_filled_df.isnull().sum())

# Define a threshold for considering category relevance
threshold = 0.70

# Calculate the percentage of the most frequent category in each column
category_counts = df_cat.apply(lambda col: col.value_counts().max() / len(col))

# Filter columns based on the threshold
unrelevant_columns = category_counts[category_counts > threshold].index
relevant_columns = category_counts[category_counts < threshold].index
print(df_cat[unrelevant_columns].shape)
print(df_cat[relevant_columns].shape)
print(f"Numerical {df_best_num_feature.shape}")

# Iterate through irrelevant columns and display their value counts
unrelevant_col = df_cat[unrelevant_columns]
relevant_columns = df_cat[relevant_columns]
for col in unrelevant_col:
    print(unrelevant_col[col].value_counts())


"""
3 - Feature Engineering
Feature engineering is the art of creating new features from existing ones or selecting the most relevant features for model training. In this section, we will engineer meaningful features to improve our model's performance.
"""
# Concatenate relevant_columns and df_best_num_feature to create a new DataFrame df1
df1 = pd.concat([relevant_columns, df_best_num_feature], axis=1)
df1.head()

# Create a box plot for 'SalePrice' vs 'LotShape' using plotly express
fig = px.box(df1, x='LotShape', y='SalePrice')
# Update layout for the plot
fig.update_layout(
    title='Sale Price Distribution by LotShape',
    xaxis=dict(title='LotShape'),
    yaxis=dict(title='Sale Price'),
    xaxis_tickangle=-45,
    width=800,
    height=400,
)
fig.show()

# Create box plots for numerical columns vs 'SalePrice'
for column in df1.columns:
    if column == 'GrLivArea':
        break
    fig = px.box(df1, x=df1[column], y='SalePrice')
    # Update layout for the plot
    fig.update_layout(
        title=f'Sale Price Distribution by {column}',
        xaxis=dict(title=column),
        yaxis=dict(title='Sale Price'),
        xaxis_tickangle=-45,
        width=800,
        height=400,
    )
    fig.show()
plt.tight_layout()
plt.show()

# Define a list of numerical columns of interest
Num_col = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'SalePrice']

# Create scatter plots for numerical columns vs 'SalePrice'
for column in df1.columns:
    if column in Num_col:
        fig = px.scatter(df1, x=df1[column], y='SalePrice')
        # Update layout for the plot
        fig.update_layout(
            title=f'Sale Price Distribution by {column}',
            xaxis=dict(title=column),
            yaxis=dict(title='Sale Price'),
            xaxis_tickangle=-45,
            width=800,
            height=400,
        )
        fig.show()

plt.tight_layout()
plt.show()

# Display value counts for each column in df1
for col in df1:
    print(col)
    print(df1[col].value_counts())

# Apply categorical value transformations for specific columns using OrdinalEncoder
# and assign transformed values to respective columns
# Repeat the process for various categorical columns

# Apply LabelEncoder to transform specific columns
df1['Has_VinylSd_Exterior'] = label_encoder.fit_transform(df1['Has_VinylSd_Exterior'])

# Display information about the DataFrame after transformations
df1.info()

# Apply StandardScaler for normalization on selected numerical columns
# Drop 'Has_VinylSd_Exterior' column
from sklearn.preprocessing import StandardScaler
numeric_col = df_best_num_feature.columns[:-1]
scaler = StandardScaler()
df1[numeric_col] = scaler.fit_transform(df1[numeric_col])
df1.drop(columns=['Has_VinylSd_Exterior'], inplace=True)

"""
Model Building
The heart of this project lies in building a predictive model. We will explore various regression algorithms, tune hyperparameters, and train models to predict house prices accurately.
"""

# Import necessary libraries and modules from scikit-learn for regression and evaluation
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.metrics import mean_squared_error  # Mean Squared Error metric
from sklearn.metrics import r2_score  # R-squared metric
from sklearn.metrics import mean_absolute_error  # Mean Absolute Error metric

# Assign the target variable 'SalePrice' to y and the features to x
y = df1['SalePrice']
x = df1.drop(columns=['SalePrice'], axis=1)

# Print shapes of y and x to check dimensions
print(f"Y shape: {y.shape}")
print(f"X shape: {x.shape}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create a Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Print shapes of training features and target
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Predict using the trained model on the test set
y_pred = model.predict(X_test)

# Print the shape of the test set
print(f"X_test shape: {X_test.shape}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared (R2) score
r2score = r2_score(y_test, y_pred)
print("R2 Score: {:.3F} %".format(r2score * 100))

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")



"""
Test Data
"""
# Extract the 'Id' column from the 'test' DataFrame
test.head()
Id = test['Id']

# Get the column names of the 'test' DataFrame
test.columns

# Get the numerical column names from 'df_best_num_feature'
num_col = df_best_num_feature.columns[:-1]
num_col

# Get the column names from the 'test' DataFrame
test_col = test.columns
test_col

# Find the common numerical columns between 'num_col' and 'test_col'
common_num_col = test_col.intersection(num_col)

# Create a DataFrame 'test_num_df' containing common numerical columns from 'test'
test_num_df = test[common_num_col].copy()

# Check for missing values in 'test_num_df'
test_num_df.isnull().sum()

# Fill missing values in 'test_num_df' with the mode (most frequent value)
test_num_df = test_num_df.fillna(test_num_df.mode().iloc[0])

# Check for missing values again after filling
test_num_df.isnull().sum()

# Get the relevant columns from 'df1'
relevant_columns.head()

# Get the column names from 'relevant_columns'
cat_col = relevant_columns.columns

# Find the common categorical columns between 'test_col' and 'cat_col'
common_cat_col = test_col.intersection(cat_col)

# Create a DataFrame 'test_cat_df' containing common categorical columns from 'test'
test_cat_df = test[common_cat_col].copy()

# Check for missing values in 'test_cat_df'
test_cat_df.isnull().sum()

# Fill missing values in 'test_cat_df' with the mode (most frequent value)
test_cat_df = test_cat_df.fillna(test_cat_df.mode().iloc[0])

# Check for missing values again after filling
test_cat_df.isnull().sum()

# Combine 'test_cat_df' and 'test_num_df' to create 'test_df1'
test_df1 = pd.concat([test_cat_df, test_num_df], axis=1)

# Display information about 'test_df1'
test_df1.info()

from sklearn.preprocessing import OrdinalEncoder

def transform_dataframe(df):
    # Define mapping for 'MasVnrType'
    masvnr_mapping = {'None': 'others', 'BrkFace': 'BrkFace', 'Other': 'others'}
    df['MasVnrType'] = df['MasVnrType'].map(masvnr_mapping).fillna('others')

    # Define mapping for 'LotShape'
    lotshape_mapping = {'Reg': 'Reg', 'IR1': 'IR1', 'Other': 'others'}
    df['LotShape'] = df['LotShape'].map(lotshape_mapping).fillna('others')

    # Define mapping for 'HouseStyle'
    housestyle_mapping = {'1Story': '1Story', '2Story': '2Story', '1.5Fin': '1.5Fin', 'Other': 'others'}
    df['HouseStyle'] = df['HouseStyle'].map(housestyle_mapping).fillna('others')

    # Define categories for ordinal encoding
    categories = {
        'Foundation': ["PConc", "CBlock", "BrkTil", "Slab", "Stone", "Wood"],
        'BsmtExposure': ["No", "Av", "Gd", "Mn"],
        'BsmtFinType1': ["Unf", "GLQ", "ALQ", "BLQ", "Rec", "LwQ"],
        'HeatingQC': ["Fa", "TA", "Gd", "Ex"],
        'KitchenQual': ["Fa", "TA", "Gd", "Ex"],
        'GarageType': ["Attchd", "Detchd", "BuiltIn", "Basment", "CarPort", "2Types"],
        'GarageFinish': ["Unf", "RFn", "Fin"]
    }

    # Apply ordinal encoding for each specified column
    encoder = OrdinalEncoder(categories=categories)
    for col, cats in categories.items():
        df[col] = encoder.fit_transform(df[[col]])
    
    return df


# Apply the transformation function to 'test_df1'
test_df1 = transform_dataframe(test_df1)

# Apply OrdinalEncoder to 'LotShape', 'HouseStyle', and other relevant columns
order_qual = ["Reg", "IR1", "others"]
qual_label = OrdinalEncoder(categories=[order_qual])
qual_label.fit(test_df1[["LotShape"]])
test_df1["LotShape"] = pd.DataFrame(qual_label.transform(test_df1[["LotShape"]]))
order_qual = ["1Story", "2Story", "1.5Fin", "others"]
qual_label = OrdinalEncoder(categories=[order_qual])
qual_label.fit(test_df1[["HouseStyle"]])
test_df1["HouseStyle"] = pd.DataFrame(qual_label.transform(test_df1[["HouseStyle"]]))
order_qual = ["None", "BrkFace", "others"]
qual_label = OrdinalEncoder(categories=[order_qual])
qual_label.fit(test_df1[["MasVnrType"]])
test_df1["MasVnrType"] = pd.DataFrame(qual_label.transform(test_df1[["MasVnrType"]]))

# Display the modified 'test_df1'
test_df1.head()

# Create a backup of 'test_df1'
backup = test_df1.copy()

# Define a function to standardize numerical columns in 'test_df1'
def standard_scalar_fun(df):
    numeric_col = df.columns[:-1]  # Exclude the last column if needed
    scaler = StandardScaler()
    df[numeric_col] = scaler.fit_transform(df[numeric_col])
    return df

# Apply the standardization function to 'test_num_df'
test_num_df = standard_scalar_fun(test_num_df)

# Create 'test_df2' by concatenating transformed 'test_df1' and 'test_num_df'
test_df2 = pd.concat([test_df1.iloc[:, :12], test_num_df], axis=1)
test_df2.head(2)

# Select columns from 'test_df1' that are also in 'train_col'
train_col = df1.columns[:-1]
test_df3 = test_df2[train_col]

# Encode 'BsmtQual' column using OrdinalEncoder
order_qual = ["TA", "Gd", "Ex", "Fa"]
qual_label = OrdinalEncoder(categories=[order_qual])
qual_label.fit(test_df3[["BsmtQual"]])
test_df3["BsmtQual"] = pd.DataFrame(qual_label.transform(test_df3[["BsmtQual"]]))

# Make a copy of 'test_df3' for predictions
x_test = test_df3.copy()

# Use the trained model to predict 'SalePrice' for test data
test_data_predict = model.predict(x_test)

# Round the predictions to 3 decimal places
test_data_predict = test_data_predict.round(3)

# Create a DataFrame 'results_df' with 'Id' and predicted 'SalePrice'
results_df = pd.DataFrame({'Id': Id, 'SalePrice': test_data_predict})

# Save the results to a CSV file
results_df.to_csv('house_prices_Advanced_linear_regression.csv', index=False)












