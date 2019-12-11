import numpy as np
from sgmcmc_ssm.models.garch import (
       generate_garch_data,
       GARCHParameters,
       GARCHPrior,
       GARCHHelper,
       GARCHSampler,
       )
from tqdm import tqdm
np.random.seed(12345)

# Parameters
## Define GARCH Parameters
alpha = 0.1
beta = 0.8
gamma = 0.05
tau = 0.3

log_mu, logit_phi, logit_lambduh = \
        GARCHParameters.convert_alpha_beta_gamma(alpha, beta, gamma)
LRinv = np.array([[tau**-1]])
parameters = GARCHParameters(
        log_mu=log_mu,
        logit_phi=logit_phi,
        logit_lambduh=logit_lambduh,
        LRinv=LRinv,
        )
print(parameters)

## Access elements of parameters
print(parameters.alpha)
print(parameters.beta)
print(parameters.gamma)
print(parameters.tau)

## Dimension of parameters
print(parameters.dim)

## Parameters as dict or as flattened numpy vector
print(parameters.as_dict())
print(parameters.as_vector())

print(parameters.from_dict_to_vector(parameters.as_dict()))
print(parameters.from_vector_to_dict(parameters.as_vector(), **parameters.dim))

# Generate Data
T = 200
data = generate_garch_data(T=T, parameters=parameters)

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
axes[0].plot(data['latent_vars'], '-C1', label='latent_var')
axes[0].set_ylabel('raw observations')
axes[1].plot(data['observations']**2, '.C0', label='data^2')
axes[1].plot(data['latent_vars']**2, '-C1', label='latent_var^2')
axes[1].set_ylabel('observations^2')
axes[1].legend()
axes[2].plot(np.log10(data['observations']**2), '.C0', label='log10(data^2)')
axes[2].plot(np.log10(data['latent_vars']**2), '-C1', label='log10(latent_var^2)')
axes[2].set_ylabel('log10(observations^2)')
axes[2].legend()
fig.suptitle("{0}".format(str(parameters)))

# GARCH Prior
## Default Prior
prior = GARCHPrior.generate_default_prior(**parameters.dim, var=1)

## Access Prior Parameters
print(prior.hyperparams)

## Sample from Prior
print(prior.sample_prior())


## Evaluate Log-Prior + Grad Log-Prior
print(prior.logprior(parameters=parameters))
print(prior.grad_logprior(parameters=parameters))


# GARCH Helper
## Setup Helper
helper = GARCHHelper(**parameters.dim)

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
print(helper.pf_score_estimate(data['observations'], parameters,
    N=1000, tqdm=tqdm,))
print(helper.pf_score_estimate(data['observations'], parameters,
    N=10000, tqdm=tqdm,))
print(helper.pf_score_estimate(data['observations'], parameters,
    pf='paris', N=1000, tqdm=tqdm,))

## Estimate Latent Mean and Covariance
x_mean, x_cov = helper.pf_latent_var_marginal(
        observations=data['observations'],
        parameters=parameters,
        N=1000,
        tqdm=tqdm,
        )
x_mean, x_cov = x_mean[:,0], x_cov[:, 0,0]
x_mean_100, x_cov_100 = helper.pf_latent_var_marginal(
        observations=data['observations'],
        parameters=parameters,
        N=100,
        tqdm=tqdm,
        )
x_mean_100, x_cov_100 = x_mean_100[:,0], x_cov_100[:, 0,0]

fig, ax = plt.subplots(1, 1)
ax.plot(x_mean, '-C0', label='smoothed_fit (1000)')
ax.plot(x_mean+np.sqrt(x_cov), "--C0", alpha=0.5)
ax.plot(x_mean-np.sqrt(x_cov), "--C0", alpha=0.5)
ax.plot(x_mean_100, '-C1', label='smoothed_fit (100)')
ax.plot(x_mean_100+np.sqrt(x_cov_100), "--C1", alpha=0.5)
ax.plot(x_mean_100-np.sqrt(x_cov_100), "--C1", alpha=0.5)
ax.plot(data['latent_vars'], '-C2', label='truth', alpha=0.8)
ax.plot(data['observations'], '.k', label='observations', alpha=0.5)
ax.legend()

## Estimate Latent Squared Mean and Covariance
x_mean, x_cov = helper.pf_latent_var_marginal(
        observations=data['observations'],
        parameters=parameters,
        N=1000,
        tqdm=tqdm,
        squared=True,
        )
x_mean, x_cov = x_mean[:,0], x_cov[:, 0,0]
x_mean_100, x_cov_100 = helper.pf_latent_var_marginal(
        observations=data['observations'],
        parameters=parameters,
        N=100,
        tqdm=tqdm,
        squared=True,
        )
x_mean_100, x_cov_100 = x_mean_100[:,0], x_cov_100[:, 0,0]

fig, ax = plt.subplots(1, 1)
ax.plot(x_mean, '-C0', label='smoothed_fit^2 (1000)')
ax.plot(x_mean+np.sqrt(x_cov), "--C0", alpha=0.5)
ax.plot(x_mean-np.sqrt(x_cov), "--C0", alpha=0.5)
ax.plot(x_mean_100, '-C1', label='smoothed_fit^2 (100)')
ax.plot(x_mean_100+np.sqrt(x_cov_100), "--C1", alpha=0.5)
ax.plot(x_mean_100-np.sqrt(x_cov_100), "--C1", alpha=0.5)
ax.plot(data['latent_vars']**2, '-C2', label='truth^2', alpha=0.8)
ax.plot(data['observations']**2, '.k', label='observations^2', alpha=0.5)
ax.legend()


# GARCH Sampler
## Setup Sampler
sampler = GARCHSampler(**parameters.dim)
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
                parameter_names=['alpha', 'beta', 'gamma', 'tau'],
                target_values=[parameters.alpha, parameters.beta,
                    parameters.gamma, parameters.tau],
                metric_names = ['mse', 'mse', 'mse', 'mse'],
                )
        ]

sample_functions = sample_function_parameters(
        ['alpha', 'beta', 'gamma', 'tau'],
        )

sampler = GARCHSampler(**parameters.dim)
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




