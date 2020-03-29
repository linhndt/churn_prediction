import pandas as pd
import os
from src.feature_engineering import one_hot_encoding


def read_csv_file(file):
    """
    Read csv file from data source

    Parameters
    --------------

    file: csv file
        A .csv file contains data (training, testing, data_dict).

    Return
    --------------

    data: data frame
        A data frame is read from csv file

    """
    if os.path.isfile(file):
        data = pd.read_csv(file)

    else:
        raise ValueError('Retry entering your path')

    return data


def add_col_name(data, unnamed_col_name="Unnamed: 30", new_col_name="dataset_name"):
    """
    Add a col name for the last col of full dataset which specifies trainining/ scoring.

    Parameters
    --------------

    data: data frame
        A full data set

    Return
    --------------

    new_data: data frame
        A data frame with the last column was named.

    """

    new_data = data.rename(columns={unnamed_col_name: new_col_name}, inplace=True)

    return new_data


def get_type_var(data_dict, type_col_name="Type", var_col_name="Variable"):
    """
    Get type of the variable from data dictionary and store them in a corresponding type list.

    Parameters
    --------------

    data_dict: data frame
        A data frame contains the information of each variable (type, meaning)

    Return
    --------------

    numeric_list, categorical_list: list
        A list stores the name of variables whose type is numeric/ categorical

    """

    numeric_list = []
    categorical_list = []

    for index, row in data_dict.iterrows():
        if row[type_col_name] == "Numeric":
            numeric_list.append(row[var_col_name])
        else:
            categorical_list.append(row[var_col_name])

    return numeric_list, categorical_list


def train_test_split(data, split_col="dataset_name_Train"):

    """
    Split original data frame into train dataset and test dataset based on split_col value (1, 0)

    Parameters
    --------------

    data: data frame
        The full data frame

    split_col:
        A column stores values (0,1), in which 1 represents train data, 0 represents test data

    Return
    --------------

    train_data, test_data: data frame
        Train and Test data frames splitted from the full data frame.

    """

    train_data = data[data[split_col] == 1].iloc[:, :-2]
    test_data = data[data[split_col] == 0].iloc[:, :-2]

    return train_data, test_data


def load_train_data(file):

    # Read csv file:
    data = read_csv_file(file)

    # Add column name:
    data = add_col_name(data)

    # One hot encode:
    data_encoded = one_hot_encoding(data)

    # Split into train, test
    train_data, test_data = train_test_split(data_encoded)

    # Split x_train, y_train:

    x_train = train_data.drop(columns=["churn", "customerID"])
    y_train = train_data[["churn"]].toarray()

    return x_train, y_train


def load_test_data(file):
    # Read csv file:
    data = read_csv_file(file)

    # Add column name:
    data = add_col_name(data)

    # One hot encode:
    data_encoded = one_hot_encoding(data)

    # Split into train, test
    train_data, test_data = train_test_split(data_encoded)

    # Split x_test, y_test:

    x_test = test_data.drop(columns=["churn", "customerID"])
    y_test = test_data[["churn"]].toarray()

    return x_test, y_test

