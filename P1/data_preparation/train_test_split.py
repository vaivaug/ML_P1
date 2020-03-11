

def get_train_test_validation_dfs(data_df):
    """ :param data_df: pandas dataframe for input data
        :return: pandas dataframes for train, test and validation data

    Split input data into test and train datasets. Split test dataset into test and validation to avoid overfitting
    """

    data_df = data_df.reset_index(drop=True)

    # keep random_state variable to always have the same test set
    # Keep 30% of the data to form test and validation sets
    test_validation = data_df.sample(frac=0.3, random_state=0)
    print('length of test and validation data together: ', len(test_validation))

    # test_validation data is split into half for test and validation sets
    test = test_validation.sample(frac=0.5, random_state=1)
    validation = test_validation.drop(test.index)

    # The other 70% of data is used for training
    train = data_df.drop(test_validation.index)
    print('length of training data: ', len(train))

    return train, test, validation
