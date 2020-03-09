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


def get_correlated_features(data_input, data_target, correlation_coef):

    correlated_features = []

    number_of_features = 0
    sum_pearsonr = 0
    sum_spearsman = 0

    for (columnName, columnData) in data_input.iteritems():
        number_of_features+=1
        corr_pearsonr, _ = pearsonr(data_input[columnName], data_target['critical_temp'])

        sum_pearsonr += abs(corr_pearsonr)

        corr_spearmanr, _ = spearmanr(data_input[columnName], data_target['critical_temp'])

        sum_spearsman += abs(corr_spearmanr)

        if (abs(corr_pearsonr) >= correlation_coef or abs(corr_spearmanr) >= correlation_coef) and columnName != "critical_temp":
            print(columnName)
            correlated_features.append(columnName)
            print(corr_pearsonr, "   ", corr_spearmanr)
            print()

    # print('average correlation: ', sum_pearsonr/number_of_features, '   ', sum_spearsman/number_of_features)
    return correlated_features



