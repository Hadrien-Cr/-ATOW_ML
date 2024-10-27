################################# LICENSING ####################################
# XGBoost Model to predict airplane TOW, ATOW ML Contest, PRC Data Challenge.
# Copyright (C) 2024 Hadrien Crassous, Eymeric Giabicani, Dariia Haryfullina, Léo le Douarec.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

#################################################################################

import sklearn
import pandas
import pickle
import os
import sys
from preprocessing.country_and_airports_codes import compute_lon_lat,group_and_rename_countries, group_and_rename_airports, group_and_rename_aircraft_types
from preprocessing.encoding import one_hot_encoding,string_to_value_count, string_to_int_hashing
from preprocessing.local_time import add_localtime_to_train_and_test
from models.xgboost_agregation import predict_tow, load_model

def main():

    ########################## GNU LICENCE #########################

    text = """
    XGBoost Model to predict airplane TOW, ATOW ML Contest, PRC Data Challenge.
    Copyright (C) 2024 Hadrien Crassous, Eymeric Giabicani, Dariia Haryfullina, Léo le Douarec.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>
    """

    print(text)


    ########################## DATA LOADING ########################

    print("-"*100)
    print("Start of the pipeline ! ")
    print("-"*100)

    train_df = pandas.read_csv('data/challenge_set.csv',index_col=0)	
    test_df = pandas.read_csv('data/submission_set.csv',index_col=0)
      
    ########################## PREPROCESSING ########################

    print("-"*100)
    print("Start of the preprocessing ! ")
    print("-"*100)

    add_localtime_to_train_and_test(train_df,test_df) #add localtime features (departures & arrival hours, day of years, weeks, month)
    compute_lon_lat(train_df, test_df) # computes lon, lat for each airport
    group_and_rename_countries(train_df, test_df) # simplify country_codes by group and rename countries
    group_and_rename_airports(train_df, test_df) # simplify airport codes by group and rename airport
    group_and_rename_aircraft_types(train_df, test_df) #regroup less used airlines and create "XXXX" category for unknown ones.


    ########################## ENCODING #########################

    print("-"*100)
    print("Start of the encoding ! ")
    print("-"*100)

    # encoding the data
    columns_to_ohe = [] # A changer
    one_hot_encoding(train_df, test_df, columns_to_ohe)

    columns_to_hash = ['callsign','country_code_ades', 'country_code_adep', 'adep', 'ades', 'airline','aircraft_type','wtc'] # A changer
    string_to_int_hashing(train_df, test_df, columns_to_hash)

    columns_to_vc = [] # A changer
    string_to_value_count(train_df, test_df, columns_to_vc)

    # drop unusefull column:
    to_drop = ['date','name_adep','name_ades','name_adep','actual_offblock_time','arrival_time','local_departure_time','local_arrival_time']
    train_df = train_df.drop(columns= to_drop)
    test_df = test_df.drop(columns= to_drop)



    ############################# MODEL #############################

    models_list = load_model("models/xgboost_agregation")

    
    ########################## PREDICT AND SAVE #####################

    print("-"*100)

    print("Start of the prediction ! ")

    y_pred = predict_tow(test_df, models_list)

    test_df['tow'] = y_pred

    submission_df = test_df[['tow']]

    submission_df.to_csv("data/results/submission_result.csv")

    print("Prediction done and saved ! ")

    pass


if __name__ == "__main__":
    main()