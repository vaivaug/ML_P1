from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math


def run_LinearRegression(train_target, train_input, test_target, test_input):

    model = LinearRegression().fit(train_input, train_target)

    predicted_temp = model.predict(test_input)
    test_target = test_target.to_numpy()

    mse = mean_squared_error(test_target, predicted_temp)
    rmse = math.sqrt(mse)
    # The mean squared error
    print('Mean squared error: ', mse)
    # The coefficient of determination: 1 is perfect prediction
    print('RMSE: ', rmse)






