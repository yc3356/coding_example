import jax
from jax._src.dtypes import dtype
import tqdm
from jax.interpreters.xla import item
from numpy.lib.function_base import cov
import pandas as pd
import numpy as np
import pyro
import numpyro
import pyro.contrib.gp as gp
import pyro.infer.mcmc as mcmc
import torch
import arviz as az
import os
import matplotlib.pyplot as plt
import pickle as pk

assert pyro.__version__.startswith('1.7.0')
pyro.set_rng_seed(0)


def plot(user_id,accuracy,X,y,est,plot_observed_data=False, plot_predictions=False, n_prior_samples=0,
         model=None, kernel=None, n_test=500):

    plt.figure(figsize=(12, 6))
    if plot_observed_data:
        incorrect = np.where(accuracy==0)
        correct = np.where(accuracy==1)
        plt.plot(temp_X.numpy()[incorrect], temp_y.numpy()[incorrect],'kx',label='incorrect')
        plt.plot(temp_X.numpy()[correct], temp_y.numpy()[correct],'ko',label='correct')
        
    if plot_predictions:
        #Xtest = torch.tensor(np.linspace(np.min(X.numpy())-0.5,np.max(X.numpy())+0.5,n_test))  # test inputs
        Xtest = temp_X
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(Xtest, full_cov=True)
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2,label='Gaussian Process')  # plot the mean
        plt.axhline(y=0, color='g', lw=1,linestyle='-')
        # plt.plot(Xtest.numpy(), IRT_pred, 'g', lw=2,label='Lognormal Model')  # plot the mean
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                         (mean - 2.0 * sd).numpy(),
                         (mean + 2.0 * sd).numpy(),
                         color='C0', alpha=0.3)
        plt.legend()
    if n_prior_samples > 0:  # plot samples from the GP prior
        #Xtest = torch.tensor(np.linspace(np.min(X.numpy())-0.5,np.max(X.numpy())+0.5,n_test))  # test inputs
        Xtest = temp_X
        noise = (model.noise if type(model) != gp.models.VariationalSparseGP
                 else model.likelihood.variance)
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = pyro.distributions.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov)\
                      .sample(sample_shape=(n_prior_samples,))
        plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)
    filename = './fit/student_' + str(user_id) + '.png'
    title = str(user_id) + ' ' + 'lengthscale: ' + str(est[0]) + '; variance: ' + str(est[1]) + '; noise: ' + str(est[2])
    plt.title(str(title))
    plt.savefig(filename)
    


data = pd.read_csv('./input.csv')
# keep = data.groupby(['question_id'])['user_id'].count()[data.groupby(['question_id'])['user_id'].count()>20].index
# data = data[data.question_id.isin(keep)]
# keep = data.groupby(['user_id'])['question_id'].count()[data.groupby(['user_id'])['question_id'].count()>20].index
# data = data[data.user_id.isin(keep)]
data = data.sort_values(['user_id','done_time'])

## find the point of new 
# users = data.user_id.unique()
# temp = [False] + ((data[data.user_id==475459076755584]['done_time'].values[1:]- data[data.user_id==475459076755584]['done_time'].values[:-1])/3600 > 1).tolist()
# np.where(temp)

# ((data.done_time[1] - data.done_time[0])/60)/60

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

user_index = np.array(user_index)
item_index = np.array(item_index)
RT = np.log(data.used_time.values)


### estimate the value of latent item and person parameters
def lognormal(I,J,user_index,item_index,RT):
    with numpyro.plate('I',I):
        tau = numpyro.sample('tau',numpyro.distributions.Normal(loc=0,scale=1))
    
    sigma_gamma = numpyro.sample('sigma_gamma',numpyro.distributions.LogNormal(loc=0,scale=1))
    mu_gamma = numpyro.sample('mu_gamma',numpyro.distributions.Normal(loc=0,scale=1))
    with numpyro.plate('J',J):
        gamma = numpyro.sample('gamma',numpyro.distributions.Normal(loc=mu_gamma,scale=sigma_gamma))
        # alpha = numpyro.sample('alpha',numpyro.distributions.LogNormal(loc=0,scale=1))
    with numpyro.plate('N',len(RT)):
        exp = gamma[item_index] - tau[user_index]
        # sigma = alpha[item_index]
        obs = numpyro.sample("obs",numpyro.distributions.Normal(loc=exp,scale=1),obs=RT)


rng_key = jax.random.PRNGKey(0)
kernel = numpyro.infer.NUTS(lognormal, max_tree_depth=10, target_accept_prob=0.8)
nuts = numpyro.infer.MCMC(kernel, num_samples=1000, num_warmup=1000)
nuts.run(rng_key,I=I,J=J,user_index=user_index,item_index=item_index,RT=RT)
# with open('./nuts.pk','wb') as handle:
#     pk.dump(nuts,handle)
samples = nuts.get_samples(group_by_chain=True)
# mcmc_summary = az.summary(samples)
tau_sumamry = az.summary(samples,var_names=['tau'])
gamma_sumamry = az.summary(samples,var_names=['gamma'])

exp_gamma = gamma_sumamry['mean'].values[item_index]
exp_tau = tau_sumamry['mean'].values[user_index]

exp_list = []
RT_list = []
RA_list = []
for u in users:
    index = np.where(data.user_id == u)[0]
    exp_list.append(exp_gamma[index]-exp_tau[index])
    RT_list.append(RT[index])
    RA_list.append(data.if_answered_correct.values[index])


for index in range(len(users)):
    temp_y = torch.tensor(RT_list[index]) - exp_list[index]
    temp_X = torch.tensor(range(len(temp_y)),dtype=torch.float64)
    index_X = torch.tensor(range(len(temp_y)),dtype=torch.long)
    pyro.clear_param_store()
    kernel = gp.kernels.RBF(input_dim=1)
    kernel.lengthscale = pyro.nn.PyroSample(pyro.distributions.Exponential(1.0))
    kernel.variance = pyro.nn.PyroSample(pyro.distributions.LogNormal(0.0, 1.0))
    gpr = gp.models.GPRegression(X=temp_X,y=temp_y, kernel=kernel)
    gpr.noise = pyro.nn.PyroSample(pyro.distributions.Exponential(1.0))
    # gpr.mean_function = lambda x: exp_list[index][index_X[int(x.item())].item()]
    
    optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    losses = []
    num_steps = 2000
    for i in tqdm.tqdm(range(num_steps)):
        optimizer.zero_grad()
        loss = loss_fn(gpr.model, gpr.guide)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    plot(model=gpr, plot_observed_data=True, plot_predictions=True,X=temp_X,
    y=temp_y,user_id=users[index],accuracy=RA_list[index],
    est=[round(kernel.lengthscale.item(),3),round(kernel.variance.item(),3),round(gpr.noise.item(),3)])


