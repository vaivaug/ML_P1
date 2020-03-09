
from prepare_data import get_clean_dataframe
from train_test_split import get_train_test_validation_dfs
from feature_selection import get_correlated_features
from LinearRegression import *
from rescaling_data import describe_data, get_train_test_normalized, get_train_test_standardized
from XGBoost import run_xgboost
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

data_filedir = "data/train.csv"

# clean data
data_df = get_clean_dataframe(data_filedir)

# split data
# do not use test set after this point till the very end
train, test, validation = get_train_test_validation_dfs(data_df)

# separate target column from the input columns
train_target = train.iloc[:, -1]
train_input = train.iloc[:, :-1]
test_target = validation.iloc[:, -1]
test_input = validation.iloc[:, :-1]


# run Linear Regression with original features
print('Linear Regression, all features, original data values: ')
run_LinearRegression(train_target, train_input, test_target, test_input)


# run Linear Regression with all features standardized
print('Linear Regression, all features, standardize data values: ')
run_LinearRegression_standardized(train_target, train_input, test_target, test_input)


# run Linear Regression with all features normalized
print('Linear Regression, all features, normalized data values: ')
run_LinearRegression_normalized(train_target, train_input, test_target, test_input)


# run Linear Regression with all features standardized and normalized
print('Linear Regression, all features, normalized and standardized data values: ')
run_LinearRegression_normalized(train_target, train_input, test_target, test_input)



# select correlated features
train_input['critical_temp'] = train_target.values
correlated_features = get_correlated_features(train_input)




scatter_matrix(train_input)
plt.show()




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