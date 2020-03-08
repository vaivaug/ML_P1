
from prepare_data import get_clean_dataframe
from train_test_split import get_train_test_validation_dfs
from feature_selection import get_correlated_features
from LinearRegression import *
from XGBoost import run_xgboost
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

# run linear regression with all features
run_LinearRegression(train_target, train_input, test_target, test_input)


# find correlated features
correlated_features = get_correlated_features(train)

# run Linear regression with correlated features
# correlated_features = get_correlated_features(train)

correlated_features = []
correlated_features.append('wtd_entropy_atomic_mass')
correlated_features.append('wtd_mean_Valence')

correlated_features = get_correlated_features(train)
# correlated_features.append('critical_temp')
train_input = train[correlated_features]
test_input = validation[correlated_features]
run_LinearRegression(train_target, train_input, test_target, test_input)

# scatter_matrix(train_input)
plt.show()

run_LinearRegression(train_target, train_input, test_target, test_input)


# run_xgboost(train, validation)

#train_target = train[['critical_temp']]
#train = train[correlated_features]

#validation_target = validation[['critical_temp']]
#validation = validation[correlated_features]



# get_predicted_values(train_target, train, validation_target, validation)

#print(len(correlated_features))



# print(train.describe())
