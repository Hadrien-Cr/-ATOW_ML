# -ATOW_ML
Team repo for ATOW ML challenge (team_zealous_watermelon).

This work has been conducted the OpenSkyNetwork PRC Data challenge, to try and predict ATOW the best way possible.
This work is being made available to anyone under the GNU General Public License v3, "giving legal permission to copy, distribute and/or modify it", as long as these freedoms are passed on.


## Quick start

```main.py``` contains all the code to run the models. We have built a pipeline that pre-processes and encodes the data, and then tunes the model parameters (XGBoost) to get the best possible results.

Another way to run models is via the jupyter notebooks that will take the same form as ```main.py```.


## Requirements 
Please make sure to put ```challenge_set.csv``` and ```submission_set.csv``` in data folder of the ATOW_ML directory. 

You need to have python version > 3.10
To install requirements please run : 

```
pip install -r requirements.txt
```


## Data preprocessing & encoding

### Preprocessing 

### Encoding 

We have chsoen to use hashing technique in order to convert data into fixed-size numerical values using a hash function (see 
```string_to_int_hasing``` function for details). 

## Algorithm choice and model selection 

We have chosen to use xgboost model 
Model parameters can be found in models/xgboost aggregation 

Modeles 

3 models xgboost - 1 spécialisé sur l'ensemble de la donnée 
2 autres - un sur H et l'autre sur M -

modele 0 sur H et modele 1 sur M 

prediction sur tous le dataset sur les 3, 
2 vecteurs de prediction reusltatnt des modeles  
combinaison linaire pour reduire les rmse - optimisation pour le faire. 

Tuning - optimisation bayseienne sur chacun des modèles (fichier tuning) module hyperopt pour ça 



