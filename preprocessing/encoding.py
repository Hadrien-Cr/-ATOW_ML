import pandas as pd 
import numpy as np
from preprocessing.country_and_airports_codes import group_and_rename_countries, group_and_rename_airports, compute_lon_lat

def one_hot_encoding(df_train, df_test, columns):
    '''
    One hot encoding for the columns in columns
    '''
    for column in columns:
        df_train[column] = df_train[column].astype('category')
        df_test[column] = df_test[column].astype('category')
        df_train = pd.concat([df_train, pd.get_dummies(df_train[column], prefix=column)], axis=1)
        df_test = pd.concat([df_test, pd.get_dummies(df_test[column], prefix=column)], axis=1)
        df_train = df_train.drop(column, axis=1)
        df_test = df_test.drop(column, axis=1)

    print("-"*100)
    print(f"Columns {columns} sucessfully one hot encoded !")
    print("-"*100)

    return df_train, df_test


def string_to_int_hashing(df_train, df_test, columns):
    '''
    String to int encoding with arbitrary hashing
    '''
    for column in columns:
        df_train[column] = df_train[column].astype('category')
        df_test[column] = df_test[column].astype('category')
        df_test[column] = df_test[column].cat.set_categories(df_train[column].cat.categories)

        df_train[column] = df_train[column].cat.codes
        df_test[column] = df_test[column].cat.codes

    print(df_test, df_train)
    print("-"*100)
    print(f"Columns {columns} successfully string to int encoded!")
    print("-"*100)

def string_to_value_count(df_train, df_test, columns):
    '''
    String to int encoding based on value_counts.
    The most frequent value gets the smallest integer (0), the second most frequent value gets the second smallest integer (1), etc.
    '''
    for column in columns:
        value_counts = df_train[column].value_counts()
        value_to_int = {value: i for i, value in enumerate(value_counts.index)}

        df_train[column] = df_train[column].map(value_to_int)
        df_test[column] = df_test[column].map(value_to_int)

    print(df_test, df_train)
    print("-" * 100)
    print(f"Columns {columns} successfully string-to-int encoded using value counts!")
    print("-" * 100)


def test_codes():
    train_df = pd.read_csv('data/challenge_set.csv')
    test_df = pd.read_csv('data/submission_set.csv')
    compute_lon_lat(train_df, test_df)
    group_and_rename_countries(train_df, test_df)
    group_and_rename_airports(train_df, test_df)
    print(train_df.columns)
    string_to_value_count(train_df, test_df, ['country_code_ades', 'country_code_adep'])
    
if __name__ == "__main__":
    test_codes()