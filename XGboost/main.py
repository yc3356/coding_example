import feature_engineer 
import xgboost_RT
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import mean_squared_error


# set up the features 
raw = pd.read_csv('./data/inputdata.csv').drop(columns=['Unnamed: 0'])
raw['done_time'] = raw['done_time'].astype(int) 
raw['used_time'] = raw['used_time'].astype(int)  
feature_engineer.Data_generation(raw)
 


# example1: reg-gamma
data = pd.read_csv('./data/data.csv').drop(columns=['Unnamed: 0'])
y = data['abs_y'] # abs_y is the used time in original scale
X = data.drop(columns=['y','abs_y'])
#### generate the traning and testing set
train_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_size)
#### this hyperparameter is what I get from the previouse hyperopt result (see tuning_gamma.py or tuning.py)
best={'colsample_bytree': 0.54, 'eta': 0.15, 'gamma': 0.96, 'max_delta_step': 5.0, 'max_depth': 7, 'min_child_weight': 10.0, 'subsample': 1.0}

xg_reg=xgb.XGBRegressor(objective='reg:gamma',**best)
evaluation = [(X_train, y_train), (X_test, y_test)] 
xg_reg.fit(X_train, y_train,eval_set=evaluation, eval_metric="rmsle",early_stopping_rounds=10,verbose=False)
pred= xg_reg.predict(X_test)
### performance
spearmanr(y_test,pred)
kendalltau(y_test,pred)
pearsonr(y_test,pred)
np.mean(abs(y_test-pred))
### visualization
result = pd.DataFrame({'predict':pred,'true':y_test})
pd.plotting.scatter_matrix(result)
plt.show()

sorted_idx = xg_reg.feature_importances_.argsort()[::-1]
plt.barh(X.columns[sorted_idx][:20], xg_reg.feature_importances_[sorted_idx][:20])
plt.xlabel("Xgboost Feature Importance")
plt.show()



# example 2: reg-logistic
data = pd.read_csv('./data/data.csv').drop(columns=['Unnamed: 0'])
y = data['y'] ## this y is the y in relative scale from 0 to 1.
X = data.drop(columns=['y','abs_y'])

train_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_size)

best={'colsample_bytree': 0.54, 'eta': 0.12, 'gamma': 0.4, 'max_delta_step': 3.0, 'max_depth': 7, 'min_child_weight': 9.0, 'subsample': 0.99}

xg_reg=xgb.XGBRegressor(objective='binary:logistic',**best)

evaluation = [(X_train, y_train), (X_test, y_test)] 
xg_reg.fit(X_train, y_train,eval_set=evaluation, eval_metric="rmsle",early_stopping_rounds=10,verbose=False)
pred= xg_reg.predict(X_test)

spearmanr(y_test,pred)
kendalltau(y_test,pred)
pearsonr(y_test,pred)
np.mean(abs(y_test-pred))

result = pd.DataFrame({'predict':pred,'true':y_test})
pd.plotting.scatter_matrix(result)
plt.show()

sorted_idx = xg_reg.feature_importances_.argsort()[::-1]
plt.barh(X.columns[sorted_idx][:20], xg_reg.feature_importances_[sorted_idx][:20])
plt.xlabel("Xgboost Feature Importance")
plt.show()