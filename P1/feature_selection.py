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


def get_correlated_features(data):

    data1 = pd.DataFrame(data, columns=['wtd_range_Valence'])
    data2 = pd.DataFrame(data, columns=['critical_temp'])
    correlated_features = []

    number_of_features=0
    sum_pearsonr = 0
    sum_spearsman = 0

    for (columnName, columnData) in data.iteritems():
        number_of_features+=1
        corr_pearsonr, _ = pearsonr(data[columnName], data['critical_temp'])
        # print('Pearsons correlation: %.3f' % corr_pearsonr)
        sum_pearsonr += abs(corr_pearsonr)

        corr_spearmanr, _ = spearmanr(data[columnName], data['critical_temp'])
        #print('Spearmans correlation: %.3f' % corr_spearmanr)
        sum_spearsman += abs(corr_spearmanr)

        if (abs(corr_pearsonr) >= 0.72 or abs(corr_spearmanr) >= 0.72) and columnName != "critical_temp":
            print(columnName)
            correlated_features.append(columnName)
            print(corr_pearsonr, "   ", corr_spearmanr)
            print()

    # print('average correlation: ', sum_pearsonr/number_of_features, '   ', sum_spearsman/number_of_features)
    return correlated_features



