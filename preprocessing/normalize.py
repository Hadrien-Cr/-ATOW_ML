import pandas as pd 
import numpy as np 

def normalisation (df_train): 
    mu = df_train['tow'].mean()
    var = df_train['tow'].var()
    std = np.sqrt(var)
    df_train['tow_normalized'] = (df_train['tow']-mu)/std
    return df_train 

#test test 

#df_train = pd.read_csv('data/challenge_set.csv')
#normalisation(df_train)
#print(df_train[['tow', 'tow_normalized']])


