from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


def print_rmse(test_target, predicted_target):
    test_target = test_target.to_numpy()

    mse = mean_squared_error(test_target, predicted_target)
    rmse = math.sqrt(mse)

    # The root mean squared error
    print('RMSE: ', rmse)


def print_mae(test_target, predicted_target):

    mae = mean_absolute_error(test_target, predicted_target)
    print('MAE: ', mae)

