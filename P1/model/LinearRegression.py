from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from data_preparation.rescaling_data import *
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from evaluation.evaluate import print_rmse, print_mae


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


def plot_test_predicted_results(test_target, predicted_target):

    fig, ax = plt.subplots()
    ax.scatter(test_target, predicted_target)
    ax.plot([test_target.min(), test_target.max()], [test_target.min(), test_target.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


def run_tests_evaluate(train_target, train_input, test_target, test_input):

    # run Linear Regression with original features
    print('Linear Regression, all features, original data values: ')
    predicted_target = get_predictions_LinearRegression(train_target, train_input, test_target, test_input)
    print_rmse(test_target, predicted_target)
    print_mae(test_target, predicted_target)

    # run Linear Regression with all features standardized
    print('Linear Regression, all features, standardize data values: ')
    predicted_target = run_LinearRegression_standardized(train_target, train_input, test_target, test_input)
    print_rmse(test_target, predicted_target)
    print_mae(test_target, predicted_target)

    # run Linear Regression with all features normalized
    print('Linear Regression, all features, normalized data values: ')
    predicted_target = run_LinearRegression_normalized(train_target, train_input, test_target, test_input)
    print_rmse(test_target, predicted_target)
    print_mae(test_target, predicted_target)

    # run Linear Regression with all features standardized and normalized
    print('Linear Regression, all features, normalized and standardized data values: ')
    predicted_target = run_LinearRegression_standardized_normalized(train_target, train_input, test_target, test_input)
    print_rmse(test_target, predicted_target)
    print_mae(test_target, predicted_target)
