import pandas as pd
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
# from tkinter import *
import numpy as np
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.feature_selection import f_regression
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest, chi2


def plot_correlating_features(data_input, data_target):
    """ :param data_input: pandas dataframe, stores feature columns of train data
        :param data_target: pandas dataframe, stores target column of train data

    Draw correlation heatmap of each feature pairs
    """

    # Using Pearson Correlation
    data_input['critical_temp'] = data_target.values
    plt.figure(figsize=(20, 10))
    cor = data_input.corr(method='pearson')
    cor = abs(cor)
    figure = plt.gcf()
    sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
    plt.show()
    plt.draw()
    figure.savefig('heatmap.png', bbox_inches='tight')
    plt.clf()


def get_correlated_features(train, cor_limit):
    """ :param data_input: pandas dataframe, stores feature columns of train data
        :param data_target: pandas dataframe, stores target column of train data
        :param cor_limit: lowest correlation fraction for a feature to still be considered 'correlating'
        :return: a list of feature names that have absolute correlation value higher than cor_limit

    Find correlation values, filter the ones where absolute value is acceptable, form a list of such feature names
    """

    cor = train.corr(method='pearson')

    # Correlation with output variable
    cor_target = abs(cor[["critical_temp"]])
    # Selecting highly correlated features
    correlated_features = cor_target[cor_target['critical_temp'] > cor_limit]
    correlated_features = correlated_features.drop("critical_temp")

    # form a list of correlating feature names, convenient format for future use
    corr_features_list = []
    print('correlating features: ')
    for row in correlated_features.index:
        corr_features_list.append(row)

    return corr_features_list

