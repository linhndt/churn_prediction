import pandas as pd
from scipy import stats


def one_hot_encoding(data):
    """
    Transform categorical data to the form of [0, ..., 1, 0, 0, ..., ]

    Parameters
    --------------

    data: data frame
        A full data set

    Return
    --------------

    data_encoded: data frame
        A transformed data frame

    """

    data_encoded = pd.get_dummies(data)

    return data_encoded


def z_score_transformation(data, numeric_list):
    """
    Transform numerical data using z score transformation (mean=0, std=1)

    Parameters
    --------------

    data: data frame
        A full data set

    numeric_list: list
        A list contains the name of the columns storing numerical values

    Return
    --------------

    transformed_data: data frame
        A transformed data frame

    """

    transformed_data = data[numeric_list].apply(stats.zscore())

    return transformed_data







