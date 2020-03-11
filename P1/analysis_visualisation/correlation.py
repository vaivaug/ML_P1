import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation(data_input, data_target):
    """ :param data_input: pandas dataframe, stores feature columns of train data
        :param data_target: pandas dataframe, stores target column of train data

    Draw correlation heatmap of each feature pairs
    """

    # Using Pearson Correlation
    data_input['critical_temp'] = data_target.values
    plt.figure(figsize=(8, 8))
    plt.xticks(rotation='vertical')
    cor = data_input.corr(method='pearson')

    sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
    plt.show()
