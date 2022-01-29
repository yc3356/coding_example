from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
from hyperopt import hp
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr

# hyperopt is a python libaray for seach space optimizing: random search and tree of parzen estimators (TPE)


data = pd.read_csv('./data/data.csv').drop(columns=['Unnamed: 0'])
y = data['y']
X = data.drop(columns=['y','abs_y'])

train_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_size)


# XGB parameters
space = {
    'learning_rate': hp.quniform('eta', 0.01,0.2, 0.01),
    'gamma': hp.quniform('gamma', 0, 1, 0.01),
    'max_depth':hp.choice('max_depth', np.arange(3, 14, dtype=int)),
    'min_child_weight':hp.quniform('min_child_weight', 1, 10, 1),        
    'max_delta_step':hp.quniform('max_delta_step', 0, 10, 1),        
    'subsample': hp.quniform('subsample', 0.5, 1, 0.01),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.01),
}



def hyperparameter_tuning(space):
    xg_reg=xgb.XGBRegressor(objective='binary:logistic',
        max_depth=space['max_depth'],learning_rate=space['learning_rate'],gamma=space['gamma'],
        min_child_weight=space['min_child_weight'],max_delta_step=int(space['max_delta_step']),
        subsample=space['subsample'],colsample_bytree=space['colsample_bytree'])

    evaluation = [(X_train, y_train), (X_test, y_test)]
    
    xg_reg.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="rmsle",
            early_stopping_rounds=10,verbose=False)

    pred= xg_reg.predict(X_test)
    pred= xg_reg.predict(X_test)
    loss_rmse = mean_squared_error(y_test, pred,squared=False)
    
    return {'loss':loss_rmse,
    'status': STATUS_OK,'model': xg_reg}


trials = Trials()
best = fmin(fn=hyperparameter_tuning,space=space,algo=tpe.suggest,max_evals=100,trials=trials)


