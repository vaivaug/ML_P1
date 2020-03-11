from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from data_preparation.rescaling_data import *


def get_predictions_LinearRegression(train_target, train_input, test_target, test_input):
    """ :param train_target: pandas dataframe, stores critical temperature column for train data
        :param train_input: pandas dataframe, stores features selected for model training
        :param test_target: pandas dataframe, stores critical temperature column for test data
        :param test_input: pandas dataframe, stores tesat data of the same features as for train_input

    Linear Regression model is trained on the input data, predictions are made
    """

    model = LinearRegression().fit(train_input, train_target)
    predicted_target = model.predict(test_input)

    return predicted_target


def run_LinearRegression_standardized(train_target, train_input, test_target, test_input):
    train_input, test_input = get_train_test_standardized(train_input, test_input)
    return get_predictions_LinearRegression(train_target, train_input, test_target, test_input)


def run_LinearRegression_normalized(train_target, train_input, test_target, test_input):
    train_input, test_input = get_train_test_normalized(train_input, test_input)
    return get_predictions_LinearRegression(train_target, train_input, test_target, test_input)


def run_LinearRegression_standardized_normalized(train_target, train_input, test_target, test_input):
    train_input, test_input = get_train_test_standardized(train_input, test_input)
    train_input, test_input = get_train_test_normalized(train_input, test_input)
    return get_predictions_LinearRegression(train_target, train_input, test_target, test_input)
