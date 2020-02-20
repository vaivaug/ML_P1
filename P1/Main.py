import numpy as np
import pandas as pd

data_df = pd.read_csv("data/train.csv")
print(data_df)

print(data_df.isnull().values.any())
print((data_df < 0).all(1) is True)

# first, split the data
# print(data_df.head())

data_m = pd.read_csv("data/unique_m.csv")
print(data_m)
