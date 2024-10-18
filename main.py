import sklearn
import pandas
import pickle
import os
import sys
from preprocessing.country_and_airports_codes import compute_lon_lat,group_and_rename_countries, group_and_rename_airports

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

    ########################## DATA SPLITTING #######################




    ############################# MODEL #############################



    
    ########################## PREDICT AND SAVE #####################



    pass


if __name__ == "__main__":
    main()