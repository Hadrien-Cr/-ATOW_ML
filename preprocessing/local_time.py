import pandas as pd
import numpy as np
import datetime

import airportsdata
import pytz
from timezonefinder import TimezoneFinder


airports = airportsdata.load("ICAO")
tf = TimezoneFinder()

def patch_kuweit(d_frame: pd.DataFrame) -> pd.DataFrame:
    try:
        d_frame[['country_code_adep','country_code_ades']]
    except:
        raise Exception(f'country_code_adep or country_code_ades column not found in dataframe:{d_frame} in function "patch kuweit')
    # replace all "##" in country_code_adep column by "KW"
    d_frame['country_code_adep'] = d_frame['country_code_adep'].replace('##', 'KW')
    d_frame['country_code_ades'] = d_frame['country_code_ades'].replace('##', 'KW')
    return d_frame


def get_airport_timezone(airport_code: str) -> str:

    airport_info = airports.get(airport_code)
    if not airport_info:
        return 'Unknown'
    lat = airport_info['lat']
    lng = airport_info['lon']

    tz_str = tf.timezone_at(lng=lng, lat=lat)
    return tz_str

def airport_tz_maps_build(d_frame: pd.DataFrame) -> dict:
    try:
        d_frame[["ades","adep"]]
    except:
        raise Exception(f'ades or adep column not found in dataframe:{d_frame} in function "airport_tz_maps_build"')
    
    ades_airport_code = d_frame["ades"].unique()
    adep_airport_code = d_frame["adep"].unique()

    union_airport_code = set(ades_airport_code).union(adep_airport_code)

    airport_tz = [get_airport_timezone(airport_code) for airport_code in union_airport_code]

    return dict(zip(union_airport_code,airport_tz))

def get_country_timezone(country_code: str) -> str:
    try: 
        return pytz.country_timezones[country_code][0]
    except:
        return "Unkown"

def country_tz_maps_build(d_frame: pd.DataFrame) -> dict:
    try:
        d_frame[['country_code_adep','country_code_ades']]
    except:
        raise Exception(f'country_code_adep or country_code_ades column not found in dataframe:{d_frame} in function "country_tz_maps_build')
    
    ades_cty_c = d_frame["country_code_ades"].unique()
    adep_cty_c = d_frame["country_code_adep"].unique()

    union_cty_code = set(ades_cty_c).union(adep_cty_c)
    cty_tz = [get_country_timezone(country_code) for country_code in union_cty_code]

    return dict(zip(union_cty_code,cty_tz))

def add_local_times(d_frame: pd.DataFrame) -> pd.DataFrame:
    try:
        d_frame[["actual_offblock_time","arrival_time"]]

    except:
        raise Exception(f'actual_offblock_time or arrival_time column not found in dataframe:{d_frame} in function "add_local_times"')
    

    airport_tz_dict = airport_tz_maps_build(d_frame)

    airport_tz_dict["EGSY"]="Europe/London"
    airport_tz_dict["OKBK"]="Asia/Kuwait"



    # add localtime based on airport to add precision
    d_frame['local_arrival_time'] = d_frame.apply(lambda x: pd.Timestamp(x['arrival_time']).tz_convert(airport_tz_dict[x['ades']]), axis=1)
    d_frame['local_departure_time'] = d_frame.apply(lambda x: pd.Timestamp(x['actual_offblock_time']).tz_convert(airport_tz_dict[x['adep']]), axis=1)

    return d_frame

def local_time_to_str(d_frame: pd.DataFrame) -> pd.DataFrame:
    try:
        d_frame[['local_arrival_time','local_departure_time']]
    except:
        raise Exception(f'local_arrival_time or local_departure_time column not found in dataframe:{d_frame} in function "local_time_to_str"')
    
    d_frame['local_arrival_time'] = d_frame['local_arrival_time'].astype(str)
    d_frame['local_departure_time'] = d_frame['local_departure_time'].astype(str)

    d_frame['local_arrival_time'] = d_frame['local_arrival_time'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    d_frame['local_departure_time']= d_frame['local_departure_time'].str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')

    return d_frame

def add_local_times_features(d_frame: pd.DataFrame) -> pd.DataFrame:
    try:
        d_frame[['local_arrival_time','local_departure_time']]
    except:
        raise Exception(f'local_arrival_time or local_departure_time column not found in dataframe:{d_frame} in function "add_local_times_features"')
    
    d_frame['local_arrival_time'] = pd.to_datetime(d_frame['local_arrival_time'])
    d_frame['local_departure_time'] = pd.to_datetime(d_frame['local_departure_time'])


    d_frame['local_arrival_hour'] = d_frame['local_arrival_time'].dt.hour
    d_frame['local_departure_hour'] = d_frame['local_departure_time'].dt.hour
    d_frame['travel_day_of_week'] = d_frame['local_arrival_time'].dt.day_of_week
    d_frame['travel_day_of_year'] = d_frame['local_arrival_time'].dt.day_of_year
    d_frame['departure_month'] = d_frame['local_arrival_time'].dt.month

    return d_frame

    
def edit_df_localtime(d_frame: pd.DataFrame) -> pd.DataFrame:
    d_frame= patch_kuweit(d_frame)
    d_frame = add_local_times(d_frame)
    d_frame= local_time_to_str(d_frame)
    d_frame= add_local_times_features(d_frame)

    return d_frame

def add_localtime_to_train_and_test(df_train: pd.DataFrame, df_test: pd.DataFrame):
    df_train = edit_df_localtime(df_train)
    df_test = edit_df_localtime(df_test)
    return df_train, df_test


def main():
    df_train = pd.read_csv("./data/submission_set.csv")
    df_train = edit_df_localtime(df_train)
    print(df_train.head())

if __name__ == "__main__":
    main()
