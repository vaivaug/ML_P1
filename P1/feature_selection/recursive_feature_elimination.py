
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


def get_best_rfe_features_LinearRegression(train_input, train_target, number_of_features):
    """ :param train_input: pandas dataframe, stores feature columns of train data
        :param train_target: pandas dataframe, stores target column of train data
        :param number_of_features: number of features we want to select
        :return: a list of selected features
    """

    model = LinearRegression()
    selector = RFE(model, number_of_features)
    selector.fit(train_input, train_target)
    sorted_features = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), list(train_input.columns)))

    # form list of best number_of_features feature names
    best_features = []
    for i in range(0, number_of_features):
        best_features.append(sorted_features[i][1])

    return best_features
