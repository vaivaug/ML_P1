from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from data_preparation.rescaling_data import *


def run_LinearRegression(train_target, train_input, test_target, test_input):
    """ :param train_target:
        :param train_input:
        :param test_target:
        :param test_input:
        :return:
    """

    model = LinearRegression().fit(train_input, train_target)

    predicted_temp = model.predict(test_input)
    test_target = test_target.to_numpy()

    mse = mean_squared_error(test_target, predicted_temp)
    rmse = math.sqrt(mse)

    # The mean squared error
    print('Mean squared error: ', mse)
    # The root mean squared error
    print('RMSE: ', rmse)


def run_LinearRegression_standardized(train_target, train_input, test_target, test_input):
    train_input, test_input = get_train_test_standardized(train_input, test_input)
    run_LinearRegression(train_target, train_input, test_target, test_input)


def run_LinearRegression_normalized(train_target, train_input, test_target, test_input):
    train_input, test_input = get_train_test_normalized(train_input, test_input)
    run_LinearRegression(train_target, train_input, test_target, test_input)


def run_LinearRegression_standardized_normalized(train_target, train_input, test_target, test_input):
    train_input, test_input = get_train_test_standardized(train_input, test_input)
    train_input, test_input = get_train_test_normalized(train_input, test_input)
    run_LinearRegression(train_target, train_input, test_target, test_input)
