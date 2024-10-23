import numpy as np
import pandas as pd

def todatetime(d_frame: pd.DataFrame) -> pd.DataFrame:
    d_frame['arrival_time'] = pd.to_datetime(d_frame['arrival_time'])
    d_frame['actual_offblock_time'] = pd.to_datetime(d_frame['actual_offblock_time'])

    return d_frame

def add_timefeature(d_frame: pd.DataFrame) -> pd.DataFrame:
    try:
        d_frame[["arrival_time","actual_offblock_time"]]
    except:
        raise ValueError("arrival_time or actual_offblock_time not found in dataframe in function 'add_timefeature'")
    
    todatetime(d_frame)
    d_frame['arrival_hour'] = d_frame['arrival_time'].dt.hour
    d_frame['departure_hour'] = d_frame['actual_offblock_time'].dt.hour
    d_frame['weekday_of_travel'] = d_frame['arrival_time'].dt.dayofweek
    d_frame['month_of_travel'] = d_frame['arrival_time'].dt.month
    d_frame['day_of_year'] = d_frame['arrival_time'].dt.dayofyear

    return d_frame
    
df_train = pd.read_csv("./data/challenge_set.csv")

df_train = add_timefeature(df_train)

print(df_train.tail(4))


