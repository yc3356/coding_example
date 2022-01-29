from jax.interpreters.xla import item
from numpy.core.fromnumeric import shape
import numpyro 
import pyro
import pandas as pd
import numpy as np
from theano.tensor.basic import tensor
import torch
from jax import random
import pymc3 as pm
import theano
import arviz as az

data = pd.read_csv('./input.csv')
keep = data.groupby(['question_id'])['user_id'].count()[data.groupby(['question_id'])['user_id'].count()>1000].index
data = data[data.question_id.isin(keep)]
keep = data.groupby(['user_id'])['question_id'].count()[data.groupby(['user_id'])['question_id'].count()>200].index
data = data[data.user_id.isin(keep)]
data = data.sort_values(['user_id','done_time'])
data.used_time[data.used_time==0] = 0.26
data.used_time[data.used_time>218] = 218

I = len(data.user_id.unique())
J = len(data.question_id.unique())


users = data.user_id.unique()
user_dict = {users[i]:i for i in range(len(users))}
user_index = [user_dict[u] for u in data.user_id]

questions = data.question_id.unique()
question_dict = {questions[i]:i for i in range(len(questions))}
item_index = [question_dict[q] for q in data.question_id]
RT = np.log(data.used_time)
RT = np.array(RT.tolist())

with pm.Model() as lognormal:
    tau = pm.Normal('tau',mu=0,sigma=1,shape=I)
    mu_gamma = pm.Normal('mu_gamm',mu=0,sigma=1)
    sigma_gamma = pm.Lognormal('sigma_gamma',mu=0,sigma=1)
    gamma = pm.Normal('gamma',mu=mu_gamma,sigma=sigma_gamma,shape=J)
    alpha = pm.Lognormal('alpha',mu=0,sigma=1,shape=J)
    obs_RT = pm.Normal('RT',mu=gamma[item_index] - tau[user_index],sigma=alpha[item_index],observed=RT)
    trace = pm.sample()

tau_summary = az.summary(trace,var_names=['tau'])
gamma_summary = az.summary(trace,var_names=['gamma'])


exp_gamma = gamma_summary['mean'].values[item_index]
exp_tau = tau_summary['mean'].values[user_index]
X_exp = exp_gamma - exp_tau 

X_exp_list = []
RT_list = []

for u in users:
    index = np.where(data.user_id == u)[0]
    temp_x_exp = X_exp[index].reshape(-1, 1)
    X_exp_list.append(temp_x_exp)
    temp_RT_exp = RT[index].reshape(-1, 1)
    RT_list.append(temp_RT_exp)


result = pd.DataFrame(columns=['lam_mean','lam_sd','eta_mean','eta_sd','sigma_mean','sigma_sd'])
for i in range(len(users)):
    temp_X = X_exp_list[i]
    temp_RT = RT_list[i]

    with pm.Model() as lognormal_PG:
        lam = pm.Exponential('lam',lam=1,shape=1)
        eta = pm.Exponential('eta',lam=1,shape=1)
        K = eta**2 * pm.gp.cov.ExpQuad(1,lam)
        sigma = pm.Lognormal('sigma',mu=0,sigma=1)
        mean = pm.gp.mean.Zero()
        gp = pm.gp.Marginal(mean_func=mean, cov_func=K)
        obs = gp.marginal_likelihood("obs", X=temp_X, y=temp_RT, noise=sigma)
        VI = pm.fit()
    trace = VI.sample()
    est = az.summary(trace)[['mean',"sd"]]
    temp_df = pd.DataFrame({'lam_mean':[est.loc['lam[0]']['mean']],'lam_sd':[est.loc['lam[0]']['sd']],
    'eta_mean':[est.loc['eta[0]']['mean']],'eta_sd':[est.loc['eta[0]']['sd']],
    'sigma_mean':[est.loc['sigma']['mean']],'sigma_sd':[est.loc['sigma']['sd']]})
    temp_df.index = [users[i]] 
    result = result.append(temp_df)

result.to_csv('result.csv')

# import matplotlib.pyplot as plt
# plt.hist(result.lam_mean)
# plt.show()