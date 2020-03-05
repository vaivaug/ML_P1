from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def get_predicted_values(train_target, train, test_target, test):

    model = LinearRegression().fit(train, train_target)

    predicted_temp = model.predict(test)
    test_target = test_target.to_numpy()
    print(len(predicted_temp))
    print(len(test_target))
    # predicted_temp = pd.DataFrame(predicted_temp, columns=['critical_temp'])

    print(test_target)
    print(predicted_temp)

    # model.predict(test_target)

    # The coefficients

    print('Coefficients: \n', model.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(test_target, predicted_temp))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(test_target, predicted_temp))

    # Plot outputs
    plt.scatter(predicted_temp, test_target, color='black')
    plt.plot(predicted_temp, test_target, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


