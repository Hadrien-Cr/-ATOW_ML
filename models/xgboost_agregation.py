# import usefull libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from preprocessing.country_and_airports_codes import compute_lon_lat,group_and_rename_countries, group_and_rename_airports, group_and_rename_aircraft_types
from preprocessing.encoding import one_hot_encoding,string_to_value_count, string_to_int_hashing
from preprocessing.local_time import add_localtime_to_train_and_test

# Load the data
def load_data(path: str) -> pd.DataFrame:
    df_set = pd.read_csv(path)
    try:
        df_set = df_set.drop(columns=['Unnamed: 0'])
    except:
        pass
    try:
        df_set[["local_departure_hour","local_arrival_hour"]]
    except:
        raise ValueError("The data is not in the right format (missing preprocessing)")
    return df_set

# train the model

def train_model(X:pd.DataFrame, y:pd.DataFrame, params: dict) -> XGBRegressor:

    model = XGBRegressor(colsample_bytree=params['colsample_bytree'],
                        gamma=params['gamma'],
                        learning_rate=params['learning_rate'],
                        max_depth=int(params['max_depth']),
                        min_child_weight=int(params['min_child_weight']),
                        n_estimators=int(params['n_estimators']),
                        reg_alpha=params['reg_alpha'],
                        reg_lambda=params['reg_lambda'],
                        subsample=params['subsample'],
                        seed=42,
                        objective="reg:squarederror",
                        #early_stopping_rounds=10,
                        eval_metric="rmse",
                        n_jobs=-1)
    
    model.fit(X, y)

    return model

def train_models(data: pd.DataFrame) -> XGBRegressor:
    X = data.drop(columns=['tow'])
    y = data['tow']

    params_set = {
        "params_wtc0": {
                        'colsample_bytree': 0.8497066984064998,
                        'gamma': 1.8567356656935454,
                        'learning_rate': 0.041891010160377044,
                        'max_depth': 9.0,
                        'min_child_weight': 8.0,
                        'n_estimators': 859.0,
                        'reg_alpha': 3.709715865940738,
                        'reg_lambda': 1.6089032820385571,
                        'subsample': 0.8005286414941816},

        "params_wtc1": {
                        'colsample_bytree': 0.9865346417369687,
                        'gamma': 2.311730004603916,
                        'learning_rate': 0.1355408626893648,
                        'max_depth': 9.0,
                        'min_child_weight': 2.0,
                        'n_estimators': 890.0,
                        'reg_alpha': 2.7683944404030436,
                        'reg_lambda': 0.10259397496527922,
                        'subsample': 0.6971029355383804
                        },

        "params_basic":{
                        'colsample_bytree': 0.9362150768343058,
                        'gamma': 2.022211195429398,
                        'learning_rate': 0.07900609044575752,
                        'max_depth': 10.0,
                        'min_child_weight': 1.0,
                        'n_estimators': 862.0,
                        'reg_alpha': 1.6659492680583492,
                        'reg_lambda': 4.4589665080717085,
                        'subsample': 0.7470405034939882,
                        
                        }
    }    


    
    model = train_model(X,y, params_set['params_basic'])


    X_wtc0 = data[data['wtc'] == 0].drop(columns=['tow'])
    y_wtc0 = data[data['wtc'] == 0]['tow']

    model_wtc0 = train_model(X_wtc0,y_wtc0, params_set['params_wtc0'])
    

    X_wtc1 = data[data['wtc'] == 1].drop(columns=['tow'])
    y_wtc1 = data[data['wtc'] == 1]['tow']

    model_wtc1 = train_model(X_wtc1,y_wtc1, params_set['params_wtc1'])
    model_wtc1.fit(X, y)

    model_list = [model_wtc0,model_wtc1,model]

    return model_list

# predict the tow

def predict_tow(data: pd.DataFrame, model_list: list) -> np.array:
    try:
        data = data.drop(columns=['Unnamed: 0'])
    except:
        pass
    try:
        data[["local_departure_hour","local_arrival_hour","lon_adep"]]
    except:
        raise ValueError("The data is not in the right format (missing preprocessing)")
    
    X = data.drop(columns=['tow'])

    y_pred_wtc = model_list[0].predict(X)*(data['wtc']==0) + model_list[1].predict(X)*(data['wtc'] == 1)

    w = [0.4951629,0.5048371]

    y_pred = w[0]*y_pred_wtc + w[1]*model_list[2].predict(X) 

    return y_pred

# evaluate the model
def evaluate_model(data: pd.DataFrame, model_list: list) -> float:
    y = data['tow']
    y_pred = predict_tow(data, model_list)
    rmse = root_mean_squared_error(y, y_pred)
    rel_error = (abs(y-y_pred)/y).mean()
    return rmse, rel_error

# save the model
def save_model(model_list: list, path: str) -> None:
    for i in range(3):
        model_list[i].save_model(f"{path}_model_{i}.json")

# load the model
def load_model(path: str) -> list:
    model_list = []
    for i in range(3):
        model = XGBRegressor()
        model.load_model(f"{path}_model_{i}.json")
        model_list.append(model)
    return model_list

# main function

def main():
    train_df = pd.read_csv('./data/challenge_set.csv',index_col=0)
    test_df = pd.read_csv('./data/submission_set.csv',index_col=0)

    # preprocessing
    add_localtime_to_train_and_test(train_df, test_df)
    compute_lon_lat(train_df, test_df)
    group_and_rename_countries(train_df, test_df)
    group_and_rename_airports(train_df, test_df)
    group_and_rename_aircraft_types(train_df, test_df)
    
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

    # train the model
    model_list = train_models(train_df)

    # save the model
    save_model(model_list, "models/xgboost_agregation")


    # predict the tow
    y_pred = predict_tow(test_df, model_list)

    # save the prediction

    test_df['tow'] = y_pred

    submission_df = test_df[['tow']]
    submission_df.to_csv("data/results/submission.csv")
    print(test_df)
    
if __name__ == "__main__":
    main()