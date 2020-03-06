import numpy as np
import pandas as pd
from prepare_data import get_clean_dataframe
from train_test_split import get_train_test_validation_dfs
from feature_selection import get_correlated_features
from LinearRegression import *
from XGBoost import run_xgboost

filedir_train = "data/train.csv"

# clean data
data_df = get_clean_dataframe(filedir_train)

# split data
# do not use test set after this point till the very end
train, test, validation = get_train_test_validation_dfs(data_df)

run_xgboost(train, test)

# analyse correlating features when visualising data
# scatter_matrix
correlated_features = get_correlated_features(train)

train_target = train[['critical_temp']]
train = train[correlated_features]

validation_target = validation[['critical_temp']]
validation = validation[correlated_features]

#get_predicted_values(train_target, train, validation_target, validation)

print(len(correlated_features))



# print(data_df)

# print(data_df.isnull().values.any())
# print((data_df < 0).all(1) is True)

# first, split the data
# print(data_df.head())

