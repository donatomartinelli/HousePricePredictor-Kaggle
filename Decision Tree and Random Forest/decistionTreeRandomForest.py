# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
#
# Load the Melbourne housing data
melbourne_file_path = 'C:/Users/donat/Desktop/melb_data.csv'
data = pd.read_csv(melbourne_file_path) 

# Separate the target (y) and predictors (X)
y = data.Price
predictors = data.drop(['Price'], axis=1)

# Split predictors into numerical and categorical features
X_numerical = predictors.select_dtypes(exclude=['object'])
X_categorical = predictors.select_dtypes('object')

# Identify categorical columns
variables = (X_categorical.dtypes == 'object')
object_cols = list(variables[variables].index)

# Calculate cardinality for each categorical variable
var_card = []
for variable in object_cols:
    unique_values = X_categorical[variable].unique()
    cardinality = len(unique_values)
    var_card.append(cardinality)

# Create a DataFrame to display variable cardinality
df = pd.DataFrame({'Variable': [var for var in object_cols], 'Cardinality': [car for car in var_card]})

# Define columns to one-hot encode
categorical_cols_to_one_hot_encode = ['Type', 'Method', 'Regionname']
X_categorical_to_encode = X_categorical[categorical_cols_to_one_hot_encode]

# One-hot encode the selected categorical columns
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(X_categorical_to_encode))
OH_cols.index = X_categorical_to_encode.index
OH_cols.columns = OH_cols.columns.astype(str)

# Drop original categorical columns that were one-hot encoded
X_categorical = X_categorical.drop(categorical_cols_to_one_hot_encode, axis=1)

# Concatenate one-hot encoded columns with the original categorical columns
X_categorical_encoded = pd.concat([X_categorical, OH_cols], axis=1)
X_categorical_encoded.columns = X_categorical_encoded.columns.astype(str)

# Define columns to perform frequency encoding
categorical_cols_to_frequency_encode = ['Suburb', 'Address', 'SellerG', 'Date', 'CouncilArea']

# Perform frequency encoding on selected categorical columns
for col in categorical_cols_to_frequency_encode:
    freq = X_categorical[col].value_counts(normalize=True)
    X_categorical[col+'_freq'] = X_categorical[col].map(freq)

# Drop original categorical columns used for frequency encoding
X_categorical = X_categorical.drop(categorical_cols_to_frequency_encode, axis=1)

# Concatenate one-hot encoded columns with the original categorical columns
X_categorical_encoded = pd.concat([X_categorical, OH_cols], axis=1)

# Concatenate one-hot encoded and numerical features
X = pd.concat([X_categorical_encoded, X_numerical], axis=1)

# Display the merged DataFrame
print(X)

# Count missing values in each column
missing_val_count_by_column = (X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Identify columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# Make copies to avoid changing the original data during imputation
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Create new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Impute missing values
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

# Define a function to calculate Mean Absolute Error (MAE) for Random Forest Regressor
def get_mae_rf(n_estimators, imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    model.fit(imputed_X_train_plus, y_train)
    preds_val = model.predict(imputed_X_valid_plus)
    mae = mean_absolute_error(y_valid, preds_val)
    return mae
# Test different values of n_estimators and print their MAE
print("Testing different n_estimators:")
estimators, mae_rf = [], []
number_of_estimators = [10, 50, 100, 200]
print("Running tests for optimal number of estimators")
for n_estimators in number_of_estimators:
    my_mae = get_mae_rf(n_estimators, imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)
    estimators.append(n_estimators)
    #print("n_estimators: %d  \t\t Mean Absolute Error: %d" % (n_estimators, my_mae))
    mae_rf.append(my_mae)
# Create a dictionary to store n_estimators as keys and their corresponding MAEs as values
pairs = {estimators[i]: mae_rf[i] for i in range(len(estimators))}
# Find the key with the lowest value (i.e., the best n_estimators value) using min() and a custom key function
optimal_n_estimators = min(pairs, key=pairs.get)
print("Optimal n_estimators:", optimal_n_estimators)
# Train the model and make predictions
model = RandomForestRegressor(n_estimators = optimal_n_estimators, random_state=0)
model.fit(imputed_X_train_plus, y_train)
randomForestPredictions = model.predict(imputed_X_valid_plus)
mae_rf = mean_absolute_error(y_valid, randomForestPredictions)
print(f"Random Forest MAE: {mae_rf}")

# Define a function to calculate Mean Absolute Error (MAE) for Decision Tree Regressor
def get_mae_dt(max_leaf_nodes, imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(imputed_X_train_plus, y_train)
    preds_val = model.predict(imputed_X_valid_plus)
    mae = mean_absolute_error(y_valid, preds_val)
    return(mae)
# Test different values of max_leaf_nodes and print their MAE
print("Testing different max_leaf_nodes:")
nodes, mae_dt = [], []
number_of_leaf_nodes = [5, 50, 500, 5000]
for max_leaf_nodes in number_of_leaf_nodes:
    my_mae = get_mae_dt(max_leaf_nodes, imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)
    nodes.append(max_leaf_nodes)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    mae_dt.append(my_mae)
# Create a dictionary to store max_leaf_nodes as keys and their corresponding MAEs as values
pairs = {nodes[i]: mae_dt[i] for i in range(len(nodes))}
# Find the key with the lowest value (i.e., the best max_leaf_nodes value) using min() and a custom key function
key_with_lowest_mae = min(pairs, key=pairs.get)
print("Key with lowest MAE:", key_with_lowest_mae)
# Extract keys and values from the dictionary
keys, values = list(pairs.keys()), list(pairs.values())
# Decision Tree 
melbourne_model = DecisionTreeRegressor(max_leaf_nodes=key_with_lowest_mae, random_state=1)
melbourne_model.fit(imputed_X_train_plus, y_train)
decisionTreePredictions = melbourne_model.predict(imputed_X_valid_plus)
mae_dt = mean_absolute_error(y_valid, decisionTreePredictions)
print(f"Decision Tree MAE: {mae_dt}")

# Create a DataFrame to compare actual vs. predicted results
comparison_df = pd.DataFrame({'Actual': y_valid,
                               'RandomForestPredictions': randomForestPredictions,
                               'DecisionTreePrediction': decisionTreePredictions})
comparison_df['RandomForestPredictions'] = comparison_df['RandomForestPredictions'].round(2)
comparison_df['DecisionTreePrediction'] = comparison_df['DecisionTreePrediction'].round(2)

# Present the model and the results
print("Comparison of Actual vs. Predicted Results:\n", comparison_df.head())
