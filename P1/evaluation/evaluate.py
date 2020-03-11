from sklearn.metrics import mean_squared_error
import math


def get_rmse(test_target, predicted_target):
    test_target = test_target.to_numpy()

    mse = mean_squared_error(test_target, predicted_target)
    rmse = math.sqrt(mse)

    # The root mean squared error
    print('RMSE: ', rmse)

    return rmse
