import numpy as np
import pandas as pd
from prepare_data import get_clean_dataframe
from train_test_split import get_train_test_dfs
from correlation import *

filedir_train = "data/train.csv"

# clean data
data_df = get_clean_dataframe(filedir_train)

# split data
# do not use test set after this point till the very end
train_df, test_df = get_train_test_dfs(data_df)

# analyse correlating features when visualising data
# scatter_matrix
draw_data(train_df)



# print(data_df)

# print(data_df.isnull().values.any())
# print((data_df < 0).all(1) is True)

# first, split the data
# print(data_df.head())

