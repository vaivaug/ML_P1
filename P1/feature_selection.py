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

    for (columnName, columnData) in data.iteritems():

        corr_pearsonr, _ = pearsonr(data[columnName], data['critical_temp'])
       # print('Pearsons correlation: %.3f' % corr_pearsonr)

        corr_spearmanr, _ = spearmanr(data[columnName], data['critical_temp'])
        #print('Spearmans correlation: %.3f' % corr_spearmanr)

        if (abs(corr_pearsonr) >= 0.7 or abs(corr_spearmanr) >= 0.7) and columnName != "critical_temp":
            print(columnName)
            correlated_features.append(columnName)
            print(corr_pearsonr, "   ", corr_spearmanr)
            print()

    return correlated_features



