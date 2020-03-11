
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


def get_best_rfe_features_LinearRegression(train_input, train_target, number_of_features):
    """ :param train_input:
        :param train_target:
        :param number_of_features:
        :return:
    """

    model = LinearRegression()
    selector = RFE(model, number_of_features, step=1)
    selector.fit(train_input, train_target)
    sorted_features = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), list(train_input.columns)))

    # form list of best number_of_features feature names
    best_features = []
    for i in range(0, number_of_features):
        best_features.append(sorted_features[i][1])

    print(best_features)

    return best_features
