import sklearn
import pandas
import pickle
import os
import sys
from preprocessing.country_and_airports_codes import compute_lon_lat,group_and_rename_countries, group_and_rename_airports, group_and_rename_aircraft_types
from preprocessing.encoding import one_hot_encoding,string_to_value_count, string_to_int_hashing
from preprocessing.local_time import add_localtime_to_train_and_test

def main():
    ########################## DATA LOADING ########################

    print("-"*100)
    print("Start of the pipeline ! ")
    print("-"*100)

    train_df = pandas.read_csv('data/challenge_set.csv')	
    test_df = pandas.read_csv('data/submission_set.csv')
      
    ########################## PREPROCESSING ########################

    print("-"*100)
    print("Start of the preprocessing ! ")
    print("-"*100)

    compute_lon_lat(train_df, test_df) # computes lon, lat for each airport
    group_and_rename_countries(train_df, test_df) # simplify country_codes by group and rename countries
    group_and_rename_airports(train_df, test_df) # simplify airport codes by group and rename airport
    group_and_rename_aircraft_types(train_df, test_df) #regroup less used airlines and create "XXXX" category for unknown ones.
    add_localtime_to_train_and_test(train_df,test_df) #add localtime features (departures & arrival hours, day of years, weeks, month)


    ########################## ENCODING #########################

    print("-"*100)
    print("Start of the encoding ! ")
    print("-"*100)

    columns_to_ohe = ['aircraft_type'] # A changer
    one_hot_encoding(train_df, test_df, columns_to_ohe)

    columns_to_hash = [] # A changer
    string_to_int_hashing(train_df, test_df, columns_to_hash)

    columns_to_vc = ['country_code_ades', 'country_code_adep', 'adep', 'ades', 'airline'] # A changer
    string_to_value_count(train_df, test_df, columns_to_vc)

    ########################## DATA SPLITTING #######################




    ############################# MODEL #############################



    
    ########################## PREDICT AND SAVE #####################



    pass


if __name__ == "__main__":
    main()