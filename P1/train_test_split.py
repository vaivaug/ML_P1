
'''function returns the same train and test sets each time the program is run'''
def get_train_test_dfs(data_df):

    data_df = data_df.reset_index(drop=True)
    # keep random_state variable to always have the same test set
    train_df = data_df.sample(frac=0.8, random_state = 0)
    test_df = data_df.drop(train_df.index)
    return train_df, test_df