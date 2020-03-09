from sklearn import preprocessing
from pandas import DataFrame


def get_train_test_normalized(data_train, data_test, norm='l1'):
    normalized_train = preprocessing.normalize(data_train, norm=norm)
    normalized_test = preprocessing.normalize(data_test, norm=norm)
    return DataFrame(normalized_train), DataFrame(normalized_test)


def get_train_test_standardized(data_train, data_test):
    standardized_train = preprocessing.scale(data_train)
    standardized_test = preprocessing.scale(data_test)
    return DataFrame(standardized_train), DataFrame(standardized_test)


def describe_data(data):
    print(data.describe())

