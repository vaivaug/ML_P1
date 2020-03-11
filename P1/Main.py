
from data_preparation.prepare_data import get_clean_dataframe
from data_preparation.train_test_split import get_train_test_validation_dfs
from model.LinearRegression import *
from analysis_visualisation.data_analysis import *
from feature_selection.recursive_feature_elimination import get_best_rfe_features_LinearRegression
from analysis_visualisation.correlation import plot_correlation_all_features
from feature_selection.Pearson_correlation import get_correlated_features, plot_correlating_features

data_filedir = "data/train.csv"

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
correlation_limit = 0.65
correlated_features = get_correlated_features(train, correlation_limit)
plot_correlating_features(train[correlated_features], train_target)


# select k best features for Linear Regression model
number_of_features = 5
best_features = get_best_rfe_features_LinearRegression(train_input, train_target, number_of_features)
print(best_features)
plot_correlating_features(train[best_features], train_target)



# run Linear Regression with original features
print('Linear Regression, all features, original data values: ')
predicted_target = get_predictions_LinearRegression(train_target, train_input, test_target, test_input)


# run Linear Regression with all features standardized
print('Linear Regression, all features, standardize data values: ')
predicted_target = run_LinearRegression_standardized(train_target, train_input, test_target, test_input)


# run Linear Regression with all features normalized
print('Linear Regression, all features, normalized data values: ')
predicted_target = run_LinearRegression_normalized(train_target, train_input, test_target, test_input)


# run Linear Regression with all features standardized and normalized
print('Linear Regression, all features, normalized and standardized data values: ')
predicted_target = run_LinearRegression_standardized_normalized(train_target, train_input, test_target, test_input)


# select correlated features
#train_input['critical_temp'] = train_target.values
#correlated_features = get_correlated_features(train_input, train_target, 0.65)

best_features = get_best_rfe_features_LinearRegression(train_input, train_target, 5)
predicted_target = get_predictions_LinearRegression(train_target, train_input[best_features], test_target, test_input[best_features])


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