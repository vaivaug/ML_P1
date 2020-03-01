import pandas as pd
import matplotlib.pyplot as plt
# from tkinter import *
import numpy as np
from pandas.plotting import scatter_matrix
import seaborn as sns


def draw_data(data):
    print('here')
    data_selected = pd.DataFrame(data, columns=['wtd_range_Valence',
                                                'std_Valence',
                                                'wtd_std_Valence',
                                                'critical_temp'])
    # scatter_matrix(data_selected, alpha=0.2, figsize=(6, 6), diagonal='kde')

    corr = data.corr()
    sns.heatmap(corr)
    plt.show()
    # df_to_plot = data_df['mean_atomic_mass', 'critical_temp']
    # pd.scatter_matrix(data_df[['mean_atomic_mass', 'critical_temp']], alpha=0.2)

