import numpy as np
from sgmcmc_ssm.models.lgssm import (
       generate_lgssm_data,
       LGSSMParameters,
       LGSSMPrior,
       LGSSMPreconditioner,
       LGSSMHelper,
       LGSSMSampler,
       )
np.random.seed(12345)

# Parameters
## Define LGSSM Parameters
A = np.eye(2)*0.9
Q = np.eye(2)*0.1
C = np.eye(2)
R = np.eye(2)*0.5

LQinv = np.linalg.cholesky(np.linalg.inv(Q))
LRinv = np.linalg.cholesky(np.linalg.inv(R))
parameters = LGSSMParameters(A=A, C=C, LQinv=LQinv, LRinv=LRinv)
print(parameters)

## Access elements of parameters
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
data = generate_lgssm_data(T=1000, parameters=parameters)

## Synthetic Data Overview
print(data.keys())
print(data['observations'])
print(data['latent_vars'])
print(data['parameters'])

## Plot Data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

fig, axes = plt.subplots(2, 2, sharex=True)
axes[0,0].plot(data['observations'][:,0], '-C0', label='observation')
axes[0,0].plot(data['latent_vars'][:,0], '--C1', label='latent')
axes[0,1].plot(data['observations'][:,1], '-C0', label='observation')
axes[0,1].plot(data['latent_vars'][:,1], '--C1', label='latent')
axes[0,1].legend()
axes[1,0].plot(data['observations'][:,0] - data['latent_vars'][:,0], '-C2', label='residual')
axes[1,1].plot(data['observations'][:,1] - data['latent_vars'][:,1], '-C2', label='residual')
axes[1,1].legend()



# LGSSM Prior
## Default Prior
prior = LGSSMPrior.generate_default_prior(**parameters.dim, var=1)

## Access Prior Parameters
print(prior.hyperparams)

## Sample from Prior
print(prior.sample_prior())


## Evaluate Log-Prior + Grad Log-Prior
print(prior.logprior(parameters=parameters))
print(prior.grad_logprior(parameters=parameters))


# LGSSM Helper
## Setup Helper
helper = LGSSMHelper(**parameters.dim)

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
xhat = helper.latent_var_sample(data['observations'], parameters)
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(data['latent_vars'][:,0], 'C0', label='truth')
axes[0].plot(xhat[:,0], ':C1', label='inferred')
axes[1].plot(data['latent_vars'][:,1], 'C0', label='truth')
axes[1].plot(xhat[:,1], ':C1', label='inferred')
axes[1].legend()

### Sample latent variables from filtered/predictive distribution
print(helper.latent_var_sample(data['observations'], parameters, distribution="filtered"))
print(helper.latent_var_sample(data['observations'], parameters, distribution="predictive"))

# LGSSM Preconditioner
preconditioner = LGSSMPreconditioner()
grad = helper.gradient_marginal_loglikelihood(data['observations'], parameters)
## Precondition Gradient
print(grad)
print(preconditioner.precondition(grad, parameters))
## Preconditioned Noise + Correction term
print(preconditioner.precondition_noise(parameters))
print(preconditioner.correction_term(parameters))

# LGSSM Sampler
## Setup Sampler
sampler = LGSSMSampler(**parameters.dim)
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
print(sampler.parameters)
for _ in range(5):
    print(sampler.sample_sgrld(epsilon=0.1, preconditioner=preconditioner).project_parameters())


## Using Evaluator
from sgmcmc_ssm import SamplerEvaluator
from sgmcmc_ssm.metric_functions import (
        sample_function_parameters,
        noisy_logjoint_loglike_metric,
        metric_function_parameters,
        )

metric_functions = [
        noisy_logjoint_loglike_metric(),
        metric_function_parameters(
                parameter_names=['A', 'Q', 'C', 'R'],
                target_values=[parameters.A, parameters.Q,
                    parameters.C, parameters.R],
                metric_names = ['mse', 'mse', 'mse', 'mse'],
                )
        ]

sample_functions = sample_function_parameters(
        ['A', 'Q', 'LQinv', 'C', 'R', 'LRinv'],
        )

sampler = LGSSMSampler(**parameters.dim)
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




