import numpy as np
from sgmcmc_ssm.models.svm import (
       generate_svm_data,
       SVMParameters,
       SVMPrior,
       SVMHelper,
       SVMSampler,
       )
from tqdm import tqdm
np.random.seed(12345)

# Parameters
## Define SVM Parameters
A = np.eye(1)*0.95
Q = np.eye(1)*1.0
R = np.eye(1)*1.0

LQinv = np.linalg.cholesky(np.linalg.inv(Q))
LRinv = np.linalg.cholesky(np.linalg.inv(R))
parameters = SVMParameters(A=A, LQinv=LQinv, LRinv=LRinv)
print(parameters)

## Access elements of parameters
print(parameters.A)
print(parameters.Q)
print(parameters.R)

## Dimension of parameters
print(parameters.dim)

## Parameters as dict or as flattened numpy vector
print(parameters.as_dict())
print(parameters.as_vector())

print(parameters.from_dict_to_vector(parameters.as_dict()))
print(parameters.from_vector_to_dict(parameters.as_vector(), **parameters.dim))

# Generate Data
T = 200
data = generate_svm_data(T=T, parameters=parameters)

## Synthetic Data Overview
print(data.keys())
print(data['observations'])
print(data['latent_vars'])
print(data['parameters'])

## Plot Data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].plot(data['observations'], '.C0', label='data')
axes[0].set_ylabel('raw observations')
axes[1].plot(data['observations']**2, '.C0', label='data^2')
axes[1].plot(parameters.R*np.exp(data['latent_vars']), '-C1', label='R*exp(latent_var)')
axes[1].set_ylabel('observations^2')
axes[1].legend()
axes[2].plot(np.log(data['observations']**2)-np.log(parameters.R), '.C0', label='log(data^2) - log(R)')
axes[2].plot(data['latent_vars'], '-C1', label='latent_var')
axes[2].set_ylabel('log(observations^2)')
axes[2].legend()
fig.suptitle("Params A={0}, Q={1}, R={2}".format(A[0,0], Q[0,0], R[0,0]))

# SVM Prior
## Default Prior
prior = SVMPrior.generate_default_prior(**parameters.dim, var=1)

## Access Prior Parameters
print(prior.hyperparams)

## Sample from Prior
print(prior.sample_prior())


## Evaluate Log-Prior + Grad Log-Prior
print(prior.logprior(parameters=parameters))
print(prior.grad_logprior(parameters=parameters))


# SVM Helper
## Setup Helper
helper = SVMHelper(**parameters.dim)

prior_var = helper.default_forward_message['precision'][0,0]**-1
prior_mean = helper.default_forward_message['mean_precision'][0] * prior_var

## Evaluate Marginal Log-likelihood using different particle filters
print(helper.pf_loglikelihood_estimate(data['observations'], parameters,
    N=100, tqdm=tqdm,))
print(helper.pf_loglikelihood_estimate(data['observations'], parameters,
    N=1000, tqdm=tqdm,))
print(helper.pf_loglikelihood_estimate(data['observations'], parameters,
    N=10000, tqdm=tqdm,))
print(helper.pf_loglikelihood_estimate(data['observations'], parameters,
    pf='paris', N=100, tqdm=tqdm))

## Evaluate Predictive Log-likelihood
print(helper.pf_predictive_loglikelihood_estimate(
    data['observations'], parameters, num_steps_ahead=10,
    N=1000, tqdm=tqdm,))

## Evaluate Gradient (Score)
print(helper.pf_gradient_estimate(data['observations'], parameters,
    N=1000, tqdm=tqdm,))
print(helper.pf_gradient_estimate(data['observations'], parameters,
    N=10000, tqdm=tqdm,))
print(helper.pf_gradient_estimate(data['observations'], parameters,
    pf='paris', N=1000, tqdm=tqdm,))

## Estimate Latent Mean and Covariance
def compare_smoothed_pfs(list_of_kwargs):
    means_covs = {}
    for kwargs in list_of_kwargs:
        name = '{0} {1}'.format(kwargs.get('pf','Poyiadjis O(N)'),
                kwargs.get('N'))
        x_mean, x_cov = helper.pf_latent_var_distr(
                observations=data['observations'],
                parameters=parameters,
                tqdm=tqdm,
                **kwargs
                )
        x_mean, x_cov = x_mean[:,0], x_cov[:, 0,0]
        means_covs[name] = x_mean, x_cov
    fig, ax = plt.subplots(1, 1)
    for ii, (name, (x_mean, x_cov)) in enumerate(means_covs.items()):
        ax.plot(x_mean, '-C{0}'.format(ii), label=name)
        ax.plot(x_mean+np.sqrt(x_cov), "--C{}".format(ii), alpha=0.5)
        ax.plot(x_mean-np.sqrt(x_cov), "--C{}".format(ii), alpha=0.5)
    ax.plot(data['latent_vars'], '-k', label='truth', alpha=0.8)
    ax.legend()
    return fig, ax

list_of_kwargs = [
        dict(N = 100),
        dict(N = 1000),
        dict(N = 10000),
        dict(pf='paris', N = 100),
        dict(pf='paris', N = 1000),
        #dict(pf='paris', N = 10000),
        ]
compare_smoothed_pfs(list_of_kwargs)

# SVM Sampler
## Setup Sampler
sampler = SVMSampler(**parameters.dim)
sampler.setup(data['observations'], prior, parameters.copy())

## Evaluate Log Joint
print(sampler.noisy_logjoint(kind='pf', return_loglike=True,
    N=1000, tqdm=tqdm))
print(sampler.noisy_logjoint(kind='pf', return_loglike=True,
    pf='paris', N=1000, tqdm=tqdm))
# Note "exact_logjoint" will throw an error
#print(sampler.exact_logjoint(return_loglike=True))

## Evaluate Gradient
### Default uses full sequence
grad = sampler.noisy_gradient(kind='pf', N=1000, tqdm=tqdm)
print(grad)

### Example with subsequence
print(sampler.noisy_gradient(kind='pf', N=1000,
    subsequence_length=10, buffer_length=5, minibatch_size=10))

## Example SGD Step
sampler.parameters = sampler.prior.sample_prior()
print(sampler.parameters)
for _ in range(5):
    print(sampler.step_sgd(kind='pf', N=1000,
        epsilon=0.1, subsequence_length=10, buffer_length=5
        ).project_parameters())

## Example ADAGRAD Step
sampler.parameters = sampler.prior.sample_prior()
print(sampler.parameters)
for _ in range(5):
    print(sampler.step_adagrad(kind='pf', N=1000,
        epsilon=0.1, subsequence_length=10, buffer_length=5
        ).project_parameters())

## Example SGLD Step
sampler.parameters = sampler.prior.sample_prior()
print(sampler.parameters)
for _ in range(5):
    print(sampler.sample_sgld(kind='pf', N=1000,
        epsilon=0.1, subsequence_length=10, buffer_length=5
        ).project_parameters())


## Using Evaluator
from sgmcmc_ssm import SamplerEvaluator
from sgmcmc_ssm.metric_functions import (
        sample_function_parameters,
        noisy_logjoint_loglike_metric,
        metric_function_parameters,
        )

metric_functions = [
        noisy_logjoint_loglike_metric(kind='pf', N=1000),
        metric_function_parameters(
                parameter_names=['A', 'LQinv', 'LRinv'],
                target_values=[parameters.A, parameters.LQinv, parameters.LRinv],
                metric_names = ['mse', 'mse', 'mse'],
                )
        ]

sample_functions = sample_function_parameters(
        ['A', 'Q', 'LQinv', 'R', 'LRinv'],
        )

sampler = SVMSampler(**parameters.dim)
sampler.setup(data['observations'], prior)
evaluator = SamplerEvaluator(
        sampler=sampler,
        metric_functions=metric_functions,
        sample_functions=sample_functions,
        )
print(evaluator.metrics)
print(evaluator.samples)

## Run a few ADA_GRAD sampler steps
for _ in tqdm(range(100)):
    evaluator.evaluate_sampler_step(
            ['step_adagrad', 'project_parameters'],
            [dict(kind='pf', N=1000,
                epsilon=0.1, subsequence_length=10, buffer_length=5), {}],
            )
print(evaluator.metrics)
print(evaluator.samples)

from sgmcmc_ssm.plotting_utils import plot_metrics, plot_trace_plot
plot_metrics(evaluator)
plot_trace_plot(evaluator)


### Run a few SGLD Steps
#for _ in range(10):
#    evaluator.evaluate_sampler_step(
#            ['sample_sgld', 'project_parameters'],
#            [dict(kind='pf', N=1000,
#                epsilon=0.1, subsequence_length=10, buffer_length=5), {}],
#            )
#print(evaluator.metrics)
#print(evaluator.samples)




