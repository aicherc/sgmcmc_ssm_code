import numpy as np
from sgmcmc_ssm.models.slds import (
       generate_slds_data,
       SLDSParameters,
       SLDSPrior,
       SLDSPreconditioner,
       SLDSHelper,
       SLDSSampler,
       )
np.random.seed(12345)

# Parameters
## Define SLDS Parameters
alpha = 0.05
delta=np.pi
b = np.pi/2

pi = np.array([
        [1-alpha, alpha],
        [alpha, 1-alpha]])
rot_mat = lambda rho: np.array([[np.cos(rho), -np.sin(rho)], [np.sin(rho), np.cos(rho)]])
A = np.array([rot_mat(-delta/2+b), rot_mat(delta/2+b)]) * 0.9
Q = np.array([np.eye(2), np.eye(2)*2]) * 0.1
C = np.eye(2)
R = np.eye(2)*0.1

logit_pi = np.log(pi+0.000001)
LQinv = np.array([np.linalg.cholesky(np.linalg.inv(Q_k)) for Q_k in Q])
LRinv = np.linalg.cholesky(np.linalg.inv(R))

parameters = SLDSParameters(logit_pi=logit_pi, A=A, C=C, LQinv=LQinv, LRinv=LRinv)
print(parameters)

## Access elements of parameters
print(parameters.pi)
print(parameters.A)
print(parameters.Q)
print(parameters.C)
print(parameters.R)

## Dimension of parameters
print(parameters.dim)

## Parameters as dict or as flattened numpy vector
print(parameters.as_dict())
print(parameters.as_vector())

print(parameters.from_dict_to_vector(parameters.as_dict()))
print(parameters.from_vector_to_dict(parameters.as_vector(), **parameters.dim))

# Generate Data
T = 1000
data = generate_slds_data(T=1000, parameters=parameters)

## Synthetic Data Overview
print(data.keys())
print(data['observations'])
print(data['latent_vars'].keys())
print(data['latent_vars']['x'])
print(data['latent_vars']['z'])
print(data['parameters'])

## Plot Data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fig, axes = plt.subplots(3, 2, sharex=True,
        gridspec_kw={'height_ratios':[6,2,2]})
axes[0,0].plot(data['observations'][:,0], '-C0', label='observation')
axes[0,0].plot(data['latent_vars']['x'][:,0], '--C1', label='latent')
axes[0,1].plot(data['observations'][:,1], '-C0', label='observation')
axes[0,1].plot(data['latent_vars']['x'][:,1], '--C1', label='latent')
axes[0,1].legend()
axes[1,0].plot(data['observations'][:,0] - data['latent_vars']['x'][:,0], '-C2', label='residual')
axes[1,1].plot(data['observations'][:,1] - data['latent_vars']['x'][:,1], '-C2', label='residual')
axes[1,1].legend()
axes[2,0].plot(data['latent_vars']['z'], '.')
axes[2,1].plot(data['latent_vars']['z'], '.')




# SLDS Prior
## Default Prior
prior = SLDSPrior.generate_default_prior(**parameters.dim, var=1)

## Access Prior Parameters
print(prior.hyperparams)

## Sample from Prior
print(prior.sample_prior())


## Evaluate Log-Prior + Grad Log-Prior
print(prior.logprior(parameters=parameters))
print(prior.grad_logprior(parameters=parameters))



# SLDS Helper
## Setup Helper
helper = SLDSHelper(**parameters.dim)
z = data['latent_vars']['z']
x = data['latent_vars']['x']

## Forward + Backward Message Passing Conditional on Z
print(helper.forward_message(data['observations'], parameters, z=z))

forward_messages = helper.forward_pass(data['observations'], parameters, z=z,
        include_init_message=True)
backward_messages = helper.backward_pass(data['observations'], parameters, z=z,
        include_init_message=True)

print(helper.marginal_loglikelihood(data['observations'], parameters, z=z))
for f_m, b_m in zip(forward_messages, backward_messages):
    print(helper.marginal_loglikelihood(np.array([]), parameters, f_m, b_m,
        z=[]))

## Forward + Backward Message Passing Conditional on X
print(helper.forward_message(data['observations'], parameters, x=x))

forward_messages = helper.forward_pass(data['observations'], parameters, x=x,
        include_init_message=True)
backward_messages = helper.backward_pass(data['observations'], parameters, x=x,
        include_init_message=True)

print(helper.marginal_loglikelihood(data['observations'], parameters, x=x))
for f_m, b_m in zip(forward_messages, backward_messages):
    print(helper.marginal_loglikelihood(np.array([]), parameters, f_m, b_m,
        x=[]))

## Complete Data Loglikelihood
print(helper.marginal_loglikelihood(data['observations'], parameters, z=z, x=x))

## Evaluate Gradient Log-likelihood
print(helper.gradient_marginal_loglikelihood(data['observations'], parameters,
    z=z))
print(helper.gradient_marginal_loglikelihood(data['observations'], parameters,
    x=x))
print(helper.gradient_marginal_loglikelihood(data['observations'], parameters,
    x=x, z=z))

## Evaluate Predictive Log-likelihood
print(helper._x_predictive_loglikelihood(data['observations'],
   z=z, parameters=parameters, lag=10))
print(helper._z_predictive_loglikelihood(data['observations'],
   x=x, parameters=parameters, lag=10))

## Gibbs Sampler Sufficient Statistic
sufficient_stat = helper.calc_gibbs_sufficient_statistic(
        data['observations'], data['latent_vars'])
print(sufficient_stat)

## Sampler parameters using Gibbs
print(helper.parameters_gibbs_sample(
    data['observations'], data['latent_vars'], prior
    ))

## Sample latent variables using Gibbs
xhat = helper._x_latent_var_sample(data['observations'], z, parameters)
xhat = helper._x_latent_var_sample(data['observations'], z, parameters, distribution='filtered')
xhat = helper._x_latent_var_sample(data['observations'], z, parameters, distribution='predictive')

zhat = helper._z_latent_var_sample(data['observations'], x, parameters)
zhat = helper._z_latent_var_sample(data['observations'], x, parameters, distribution='filtered')
zhat = helper._z_latent_var_sample(data['observations'], x, parameters, distribution='predictive')

### Plots for X
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(data['latent_vars']['x'][:,0], 'C0', label='truth')
axes[0].plot(xhat[:,0], ':C1', label='inferred')
axes[1].plot(data['latent_vars']['x'][:,1], 'C0', label='truth')
axes[1].plot(xhat[:,1], ':C1', label='inferred')
axes[1].legend()

### Plots for Z
fig, axes = plt.subplots(1, 1)
axes.plot(data['latent_vars']['z'], 'C0.', label='truth')
axes.plot(zhat+0.1, 'C1.', label='smoothed sample')
axes.legend()


# SLDS Preconditioner
preconditioner = SLDSPreconditioner()

parameters.pi_type = 'expanded'
grad = helper.gradient_marginal_loglikelihood(data['observations'], parameters, z=z, x=x)
## Precondition Gradient
print(grad)
print(preconditioner.precondition(grad, parameters))
## Preconditioned Noise + Correction term
print(preconditioner.precondition_noise(parameters))
print(preconditioner.correction_term(parameters))

# SLDS Sampler
## Setup Sampler
sampler = SLDSSampler(**parameters.dim)
sampler.setup(data['observations'], prior, parameters.copy())
sampler.init_sample_latent()

## Evaluate Log Joint
#print(sampler.exact_logjoint(return_loglike=True))
#print(sampler.noisy_logjoint(return_loglike=True, subsequence_length=-1))
print(sampler.noisy_logjoint(return_loglike=True, subsequence_length=50))
print(sampler.noisy_logjoint(return_loglike=True, subsequence_length=10, minibatch_size=5))

## Evaluate Gradient
### Default uses full sequence
grad = sampler.noisy_gradient()
print(grad)
### Example with subsequence
print(sampler.noisy_gradient(kind='complete', subsequence_length=10, buffer_length=5, minibatch_size=10))
print(sampler.noisy_gradient(kind='x_marginal', subsequence_length=10, buffer_length=5, minibatch_size=10))
print(sampler.noisy_gradient(kind='z_marginal', subsequence_length=10, buffer_length=5, minibatch_size=10))

## Preconditioned Gradient
precond_grad = sampler.noisy_gradient(preconditioner=preconditioner)
print(precond_grad)
### Example with subsequence
print(sampler.noisy_gradient(
    preconditioner=preconditioner,
    subsequence_length=10, buffer_length=5, minibatch_size=10))

## Example Gibbs Step
sampler.parameters = sampler.prior.sample_prior()
print(sampler.parameters)
for _ in range(5):
    print(sampler.sample_gibbs().project_parameters())

## Example SGD Step
sampler.parameters = sampler.prior.sample_prior()
print(sampler.parameters)
for _ in range(5):
    print(sampler.step_sgd(epsilon=0.1, subsequence_length=10, buffer_length=5
        ).project_parameters())

## Example ADAGRAD Step
sampler.parameters = sampler.prior.sample_prior()
print(sampler.parameters)
for _ in range(5):
    print(sampler.step_adagrad(epsilon=0.1, subsequence_length=10, buffer_length=5
        ).project_parameters())

## Example SGLD Step
sampler.parameters = sampler.prior.sample_prior()
print(sampler.parameters)
for _ in range(5):
    print(sampler.sample_sgld(epsilon=0.1).project_parameters())

## Example SGRLD Step
sampler.parameters = sampler.prior.sample_prior()
sampler.parameters.pi_type = 'expanded'
print(sampler.parameters)
for _ in range(5):
    print(sampler.sample_sgrld(epsilon=0.1, preconditioner=preconditioner).project_parameters())


## Using Evaluator
from tqdm import tqdm
from sgmcmc_ssm import SamplerEvaluator
from sgmcmc_ssm.metric_functions import (
        sample_function_parameters,
        noisy_logjoint_loglike_metric,
        metric_function_parameters,
        best_permutation_metric_function_parameter,
        best_double_permutation_metric_function_parameter,
        )
metric_functions = [
        noisy_logjoint_loglike_metric(),
        best_double_permutation_metric_function_parameter(
            parameter_name = 'pi',
            target_value = parameters.pi,
            metric_name = 'mse',
            best_function = min
            ),
        best_permutation_metric_function_parameter(
            parameter_name = 'A',
            target_value = parameters.A,
            metric_name = 'mse',
            best_function = min
            ),
        best_permutation_metric_function_parameter(
            parameter_name = 'Q',
            target_value = parameters.Q,
            metric_name = 'mse',
            best_function = min
            ),
        metric_function_parameters(
                parameter_names=['C', 'R'],
                target_values=[parameters.C, parameters.R],
                metric_names = ['mse', 'mse'],
                )
        ]

sample_functions = sample_function_parameters(
        ['pi', 'A', 'Q', 'C', 'R'],
        )

sampler = SLDSSampler(**parameters.dim)
sampler.setup(data['observations'], prior)
sampler.init_sample_latent() ## THIS IS IMPORTANT
evaluator = SamplerEvaluator(
        sampler=sampler,
        metric_functions=metric_functions,
        sample_functions=sample_functions,
        )
print(evaluator.metrics)
print(evaluator.samples)

## Run a few Gibbs Sampler steps
for _ in tqdm(range(10)):
    evaluator.evaluate_sampler_step(['sample_gibbs', 'project_parameters'])
print(evaluator.metrics)
print(evaluator.samples)

## Run a few ADA_GRAD sampler steps
for _ in tqdm(range(10)):
    evaluator.evaluate_sampler_step(
            ['step_adagrad', 'project_parameters'],
            [dict(epsilon=0.1, subsequence_length=10, buffer_length=5), {}],
            )
print(evaluator.metrics)
print(evaluator.samples)


## Run a few SGRLD Steps
evaluator.sampler.parameters.pi_type='expanded'
for _ in tqdm(range(10)):
    evaluator.evaluate_sampler_step(
            ['sample_sgrld', 'project_parameters'],
            [dict(preconditioner=preconditioner,
                epsilon=0.1, subsequence_length=10, buffer_length=5), {}],
            )
print(evaluator.metrics)
print(evaluator.samples)

from sgmcmc_ssm.plotting_utils import plot_metrics, plot_trace_plot
plot_metrics(evaluator)
plot_trace_plot(evaluator, single_variables=['C', 'LRinv', 'R', 'Rinv'])





