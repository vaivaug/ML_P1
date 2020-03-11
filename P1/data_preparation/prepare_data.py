import pandas as pd


def get_clean_dataframe(filedir):
    """
    :param filedir: String value, file path of the input data file
    :return: input data stored in pandas dataframe with all values being non negative numbers

    Read data and check dataframe values
    """

    data_df = pd.read_csv(filedir)

    # in this case, the if statement is True. Therefore 'else' statement is not needed
    if check_no_negative_values(data_df) and check_all_numeric_values(data_df):
        return data_df


def check_no_negative_values(data_df):
    """
    :param data_df: input data in pandas dataframe
    :return: boolean value, 'True' if all values are non negative

    Check that all values in the given dataframe are non negative
    """

    columns_non_negative = (data_df >= 0).all()

    for column_name, column_data in columns_non_negative.iteritems():
        if columns_non_negative[column_name] is False:
            return False

    return True


def check_all_numeric_values(data_df):
    """
    :param data_df: data in pandas dataframe
    :return: boolean value, 'True' if all values are numeric

    Check that all values in the given dataframe are numeric
    """

    for (column_name, column_data) in data_df.iteritems():
        if not pd.to_numeric(data_df[column_name], errors='coerce').notnull().all():
            return False

    return True
