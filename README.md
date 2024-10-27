# 🛫 ATOW_ML 🛬
Team repo for ATOW ML challenge (team_zealous_watermelon).

This work has been conducted the [OpenSkyNetwork PRC Data challenge](https://ansperformance.eu/study/data-challenge/), to try and predict ATOW the best way possible.
This work is being made available to anyone under the GNU General Public License v3, "giving legal permission to copy, distribute and/or modify it", as long as these freedoms are passed on.

Open Sky Network: https://opensky-network.org.

## Quick start

```main.py``` contains all the code to run the models. We have built a pipeline that pre-processes and encodes the data, and then tunes the model parameters (XGBoost) to get the best possible results.

Another way to run models is via the jupyter notebooks that take the same form as ```main.py```.


## Requirements 
⚠️ : Please make sure to put ```challenge_set.csv``` and ```submission_set.csv``` in the data folder of the ATOW_ML directory. 

You need to have python version > 3.10
To install requirements please run : 

```
pip install -r requirements.txt
```

## Data preprocessing & encoding

### Preprocessing 
We have added several adjustment to the given dataset in order to prepare data for training of XGboost model: 
1. Converted timezone to local time and adding features such as day of the year, week, month to account for eventual seasonality in ```add_localtime_to_train_and_test```
2. Computed longitude and latitude for each airport in ```compute_lon_lat```
3. Simplified country_codes by group and rename countries in ```group_and_rename_countries```
4. Simplify airport codes by group and rename airport in ```group_and_rename_airports```
5. Regrouped less used airlines and aircraft types and create "XXXX" category for unknown ones in ```group_and_rename_aircraft_types```


### Encoding 

We have chsoen to use hashing technique in order to convert data into fixed-size numerical values using a hash function (see 
```string_to_int_hasing``` function for details). 

## Algorithm choice and model selection 

We have chosen to use XGboost model for prediction of tow. 
Model parameters can be found in `"models/xgboost_aggregation.py"`

We have specified and trained three different XGboost models : 
- One on the whole dataset
- Two others on H or M wtc (xgboost_agreggation_model_0.json stands for wtc = H, xgboost_agreggation_model_1.json stands for wtc = M, xgboost_agreggation_model_2.json stands for the whole dataset)

We then combine them linearly to minimize RMSE, optimized for best performance.

Bayesian optimization was applied to each model using the Hyperopt module for hyperparameter tuning (see `tuning.py` file).




