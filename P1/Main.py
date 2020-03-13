
from data_preparation.prepare_data import get_clean_dataframe
from data_preparation.train_test_split import get_train_test_validation_dfs
from model.LinearRegression import *
from analysis_visualisation.data_analysis import *
from feature_selection.recursive_feature_elimination import get_best_rfe_features_LinearRegression
from analysis_visualisation.correlation import plot_correlation_all_features
from feature_selection.Pearson_correlation import get_correlated_features, plot_correlating_features

data_filedir = "data/train.csv"
import matplotlib.pyplot as plt
import pandas as pd


# clean data
data_df = get_clean_dataframe(data_filedir)

# split data
# do not use test set after this point till the very end
train, test, validation = get_train_test_validation_dfs(data_df)

# describe data
describe_data(train)

# separate target column from the input columns
train_target = train.iloc[:, -1]
train_input = train.iloc[:, :-1]
test_target = validation.iloc[:, -1]
test_input = validation.iloc[:, :-1]

# visualise correlation all features plot
# plot_correlation_all_features(train_input, train_target)

# select and plot correlated features Pearson correlation
correlation_limit = 0.63
correlated_features = get_correlated_features(train, correlation_limit)
# plot_correlating_features(train[correlated_features], train_target)
print('Pearson features: ', correlated_features)


# select k best features for Linear Regression model using RFE
number_of_features = 5
rfe_features = get_best_rfe_features_LinearRegression(train_input, train_target, number_of_features)
print('RFE features: ', rfe_features)
# plot_correlating_features(train[best_features], train_target)

# run Linear Regression 4 times with all features
print("Linear Regression all features: ")
run_tests_evaluate(train_target, train_input, test_target, test_input)

print("Linear Regression Pearson correlated features")
run_tests_evaluate(train_target, train_input[correlated_features], test_target, test_input[correlated_features])

print("Linear Regression RFE selected features")
run_tests_evaluate(train_target, train_input[rfe_features], test_target, test_input[rfe_features])




#scatter_matrix(train_input)
#plt.show()




'''
# run linear regression with all features
run_LinearRegression(train_target, train_input, test_target, test_input)


# find correlated features
correlated_features = get_correlated_features(train)

# run Linear regression with correlated features
train_input = train[correlated_features]
test_input = validation[correlated_features]
run_LinearRegression(train_target, train_input, test_target, test_input)


correlated_features = []
correlated_features.append('wtd_entropy_atomic_mass')
correlated_features.append('wtd_mean_Valence')

# scatter_matrix(train_input)
# plt.show()
'''

# run_xgboost(train, validation)


# print(train.describe())


'''
train_input = train.iloc[:, :-1]
test_input = validation.iloc[:, :-1]
train_input = train_input.append(test_input)


train_input, norms = normalise_data(train_input)
print('inputs normalised together')
print('length: ', len(train_input))
print(train_input)

'''