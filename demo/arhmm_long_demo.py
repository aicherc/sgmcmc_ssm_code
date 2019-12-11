import numpy as np
from sgmcmc_ssm.models.arphmm import (
       generate_arphmm_data,
       ARPHMMParameters,
       ARPHMMPrior,
       ARPHMMPreconditioner,
       ARPHMMHelper,
       ARPHMMSampler,
       )
np.random.seed(12345)

# Parameters
## Define ARPHMM Parameters
logit_pi = np.array([[2, 0], [0, 2]])*2
D = np.array([np.eye(2), -np.eye(2)]) * 0.9
R = np.array([np.eye(2), np.eye(2)]) * 0.01
LRinv = np.array([np.linalg.cholesky(np.linalg.inv(R_k)) for R_k in R])
parameters = ARPHMMParameters(logit_pi=logit_pi, D=D, LRinv=LRinv)
print(parameters)

## Access elements of parameters
print(parameters.pi)
print(parameters.D)
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
data = generate_arphmm_data(T=1000, parameters=parameters)

## Synthetic Data Overview
print(data.keys())
print(data['observations'])
print(data['latent_vars'])
print(data['parameters'])

## Plot Data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3,1]})
axes[0].plot(data['observations'][:,0,0], 'C0')
axes[0].plot(data['observations'][:,0,1], 'C1')
axes[1].plot(data['latent_vars'], '.')

# ARPHMM Prior
## Default Prior
prior = ARPHMMPrior.generate_default_prior(**parameters.dim, var=1)

## Access Prior Parameters
print(prior.hyperparams)

## Sample from Prior
print(prior.sample_prior())


## Evaluate Log-Prior + Grad Log-Prior
print(prior.logprior(parameters=parameters))
print(prior.grad_logprior(parameters=parameters))



# ARPHMM Helper
## Setup Helper
helper = ARPHMMHelper(**parameters.dim)

## Forward + Backward Message Passing
print(helper.forward_message(data['observations'], parameters))

forward_messages = helper.forward_pass(data['observations'], parameters, include_init_message=True)
backward_messages = helper.backward_pass(data['observations'], parameters, include_init_message=True)

## Evaluate Marginal Log-likelihood
print(helper.marginal_loglikelihood(data['observations'], parameters))
for f_m, b_m in zip(forward_messages, backward_messages):
    print(helper.marginal_loglikelihood(np.array([]), parameters, f_m, b_m))

## Evaluate Gradient Marginal Log-likelihood
print(helper.gradient_marginal_loglikelihood(data['observations'], parameters))

## Evaluate Predictive Log-likelihood
print(helper.predictive_loglikelihood(data['observations'], parameters, lag=10))

## Gibbs Sampler Sufficient Statistic
sufficient_stat = helper.calc_gibbs_sufficient_statistic(
        data['observations'], data['latent_vars'])
print(sufficient_stat)

## Sampler parameters using Gibbs
print(helper.parameters_gibbs_sample(
    data['observations'], data['latent_vars'], prior
    ))

## Sample latent variables using Gibbs
### Default is smoothed distribution
zhat = helper.latent_var_sample(data['observations'], parameters)
print(np.sum(zhat != data['latent_vars']))

fig, ax = plt.subplots(1, 1)
ax.plot(data['latent_vars'], 'C0.', label='truth')
ax.plot(zhat+0.1, 'C1.', label='smoothed sample')
for err_loc in np.where(zhat != data['latent_vars'])[0]:
    ax.axvline(x=err_loc, color='red', linewidth=1, linestyle='--')
ax.legend()
ax.set_xlabel("red lines are errors")

from sklearn.metrics import confusion_matrix
print('Confusion Matrix:')
print(confusion_matrix(data['latent_vars'], zhat))


### Sample latent variables from filtered/predictive distribution
print(helper.latent_var_sample(data['observations'], parameters, distribution="filtered"))
print(helper.latent_var_sample(data['observations'], parameters, distribution="predictive"))

# ARPHMM Preconditioner
preconditioner = ARPHMMPreconditioner()
parameters.pi_type = 'expanded'
grad = helper.gradient_marginal_loglikelihood(data['observations'], parameters)
## Precondition Gradient
print(grad)
print(preconditioner.precondition(grad, parameters))
## Preconditioned Noise + Correction term
print(preconditioner.precondition_noise(parameters))
print(preconditioner.correction_term(parameters))

# ARPHMM Sampler
## Setup Sampler
sampler = ARPHMMSampler(**parameters.dim)
sampler.setup(data['observations'], prior, parameters.copy())

## Evaluate Log Joint
print(sampler.exact_logjoint(return_loglike=True))

## Evaluate Gradient
### Default uses full sequence
grad = sampler.noisy_gradient()
print(grad)
### Example with subsequence
print(sampler.noisy_gradient(subsequence_length=10, buffer_length=5, minibatch_size=10))

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
from sgmcmc_ssm import SamplerEvaluator
from sgmcmc_ssm.metric_functions import (
        sample_function_parameters,
        noisy_logjoint_loglike_metric,
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
            parameter_name = 'D',
            target_value = parameters.D,
            metric_name = 'mse',
            best_function = min
            ),
        best_permutation_metric_function_parameter(
            parameter_name = 'R',
            target_value = parameters.R,
            metric_name = 'mse',
            best_function = min
            ),
        ]
sample_functions = sample_function_parameters(
        ['logit_pi', 'expanded_pi', 'pi', 'D', 'R', 'LRinv'],
        )

sampler = ARPHMMSampler(**parameters.dim)
sampler.setup(data['observations'], prior)
evaluator = SamplerEvaluator(
        sampler=sampler,
        metric_functions=metric_functions,
        sample_functions=sample_functions,
        )
print(evaluator.metrics)
print(evaluator.samples)

## Run a few Gibbs Sampler steps
for _ in range(10):
    evaluator.evaluate_sampler_step(['sample_gibbs', 'project_parameters'])
print(evaluator.metrics)
print(evaluator.samples)

## Run a few ADA_GRAD sampler steps
for _ in range(10):
    evaluator.evaluate_sampler_step(
            ['step_adagrad', 'project_parameters'],
            [dict(epsilon=0.1, subsequence_length=10, buffer_length=5), {}],
            )
print(evaluator.metrics)
print(evaluator.samples)


## Run a few SGRLD Steps
evaluator.sampler.parameters.pi_type = 'expanded'
for _ in range(10):
    evaluator.evaluate_sampler_step(
            ['sample_sgrld', 'project_parameters'],
            [dict(preconditioner=preconditioner,
                epsilon=0.1, subsequence_length=10, buffer_length=5), {}],
            )
print(evaluator.metrics)
print(evaluator.samples)

from sgmcmc_ssm.plotting_utils import plot_metrics, plot_trace_plot
plot_metrics(evaluator)
plot_trace_plot(evaluator)




