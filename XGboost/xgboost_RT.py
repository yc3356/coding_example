from datetime import time
import pandas as pd
import numpy as np
from scipy.stats.stats import RepeatedResults
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, kendalltau, pearsonr

def XGBoost():
    data = pd.read_csv('./data/data.csv').drop(columns=['Unnamed: 0'])

    y = data['y']
    X = data.drop(columns=['y'])
    data_dmatrix = xgb.DMatrix(data=X,label=y)

    train_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_size)

    xg_reg = xgb.XGBRegressor(objective='binary:logistic')

    xg_reg.fit(X_train,y_train,verbose=True)

    sorted_idx = xg_reg.feature_importances_.argsort()[::-1]
    plt.barh(X.columns[sorted_idx][:20], xg_reg.feature_importances_[sorted_idx][:20])
    plt.xlabel("Xgboost Feature Importance")
    plt.savefig('top20importantfeature.png')


    preds = xg_reg.predict(X_test)

    result = pd.DataFrame({'predict':preds,'true':y_test})
    pd.plotting.scatter_matrix(result)
    plt.savefig('scatter_matrix.png')



    a=round(mean_squared_error(y_test, preds,squared=False),3)
    b=round(spearmanr(y_test, preds)[0],3)
    c=round(pearsonr(y_test, preds)[0],3)
    d=round(kendalltau(y_test, preds)[0],3)
    return (a,b,c,d)



    