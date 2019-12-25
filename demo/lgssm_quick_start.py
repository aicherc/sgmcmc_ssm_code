import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(12357)

###############################################################################
# Generate Synthetic Data
###############################################################################
from sgmcmc_ssm.models.lgssm import (
    LGSSMParameters,
    generate_lgssm_data,
    )
T = 1000

## Parameters
A = np.array([[0.9753, -0.0961], [ 0.0961, 0.9753]])
Q = np.eye(2)*0.1
C = np.eye(2)
R = np.eye(2)*0.5
parameters = LGSSMParameters(A=A, C=C, Q=Q, R=R)

## Generate Data
data = generate_lgssm_data(T=1000, parameters=parameters)

## Plot Data
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(data['observations'][:,0], '-C0', label='observation')
axes[0].plot(data['latent_vars'][:,0], '--C1', label='latent')
axes[0].set_ylabel("dim = 0")
axes[1].plot(data['observations'][:,1], '-C0', label='observation')
axes[1].plot(data['latent_vars'][:,1], '--C1', label='latent')
axes[1].set_ylabel("dim = 1")
axes[1].set_xlabel("t")
axes[0].legend()

###############################################################################
# Setup Sampler
###############################################################################
from sgmcmc_ssm.models.lgssm import LGSSMSampler
sampler = LGSSMSampler(n=2, m=2, observations=data['observations'])

# Fit Using Gibbs
sampler.prior_init()
print(sampler.exact_logjoint())
sampler.fit(num_iters=30, iter_type='Gibbs', tqdm=tqdm)
print(sampler.exact_logjoint())

## Plot Smoothing Distr for Latent Variables
def plot_distr(mean, cov, data, xmin=0, xmax=200):
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(mean[:,0], '-C1', label='Post Mean')
    axes[0].plot(mean[:,0]+cov[:,0,0]**0.5, '--C1', label='Post Mean +/- SD')
    axes[0].plot(mean[:,0]-cov[:,0,0]**0.5, '--C1')
    axes[0].plot(data['latent_vars'][:,0], '-C0', label='Truth')
    axes[0].plot(data['observations'][:,0], 'xC2', label='Observations')
    axes[0].legend()
    axes[1].plot(mean[:,1], '-C1')
    axes[1].plot(mean[:,1]+cov[:,1,1]**0.5, '--C1')
    axes[1].plot(mean[:,1]-cov[:,1,1]**0.5, '--C1')
    axes[1].plot(data['latent_vars'][:,1], '-C0')
    axes[1].plot(data['observations'][:,1], 'xC2')
    axes[1].set_xlim(xmin, xmax)

def plot_samples(samples, data, xmin=0, xmax=200):
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(samples[:,0,:], 'C1', alpha=0.05)
    axes[0].plot(data['latent_vars'][:,0], '-C0', label='Truth')
    axes[0].plot(data['observations'][:,0], 'xC2', label='Observations')
    axes[0].legend()
    axes[1].plot(samples[:,1,:], '-C1', alpha=0.05)
    axes[1].plot(data['latent_vars'][:,1], '-C0')
    axes[1].plot(data['observations'][:,1], 'xC2')
    axes[1].set_xlim(xmin, xmax)

### Marginal Mean + Covariance
mean_x, cov_x = sampler.predict(target='latent', return_distr=True)
plot_distr(mean_x, cov_x, data)

### Samples from Posterior
xs = sampler.predict(target='latent', num_samples=100)
plot_samples(xs, data)


# Fit Using SGLD
sampler.prior_init()
print(sampler.exact_logjoint())
sampler.fit(num_iters=1000, iter_type='SGLD',
        epsilon=0.01, subsequence_length=16, buffer_length=4,
        tqdm=tqdm)
print(sampler.exact_logjoint())

# Fit Using SGRLD
sampler.prior_init()
print(sampler.exact_logjoint())
sampler.fit(num_iters=1000, iter_type='SGRLD',
        epsilon=0.01, subsequence_length=16, buffer_length=4,
        tqdm=tqdm)
print(sampler.exact_logjoint())

# Fit Using ADAGRAD
sampler.prior_init()
print(sampler.exact_logjoint())
sampler.fit(num_iters=1000, iter_type='ADAGRAD',
        epsilon=0.1, subsequence_length=16, buffer_length=4,
        tqdm=tqdm)
print(sampler.exact_logjoint())


###############################################################################
# Evaluate Sampler
###############################################################################
## Parameters to evaluate
sampler.prior_init()
parameters_list = sampler.fit(num_iters=1000, iter_type='SGRLD',
        epsilon=0.1, subsequence_length=16, buffer_length=4,
        tqdm=tqdm, output_all=True,
        )

## Specify Metric Functions
metric_functions = []

### Loglikelihood and Logjoint
from sgmcmc_ssm.metric_functions import noisy_logjoint_loglike_metric
metric_functions += [noisy_logjoint_loglike_metric()]

### log10 MSE at recovering X
from sgmcmc_ssm.metric_functions import metric_compare_x
metric_functions += [metric_compare_x(data['latent_vars'])]

### log10 MSE at recovering parameters
from sgmcmc_ssm.metric_functions import metric_function_parameters
metric_functions += [
        metric_function_parameters(
                parameter_names=['A', 'Q', 'R'],
                target_values=[parameters.A, parameters.Q, parameters.R],
                metric_names = ['logmse', 'logmse', 'logmse'],
                )
        ]

## Specify Sample Functions
from sgmcmc_ssm.metric_functions import sample_function_parameters
sample_functions = sample_function_parameters(
        ['A', 'Q', 'LQinv', 'R', 'LRinv'],
        )


# Offline Evaluation
from sgmcmc_ssm.evaluator import OfflineEvaluator
evaluator = OfflineEvaluator(
        sampler=sampler,
        parameters_list=parameters_list,
        metric_functions=metric_functions, sample_functions=sample_functions,
        )
evaluator.evaluate(num_to_eval=40, tqdm=tqdm)
print(evaluator.get_metrics())
print(evaluator.get_samples())

# Plot Results
from sgmcmc_ssm.plotting_utils import plot_metrics, plot_trace_plot
plot_metrics(evaluator, burnin=10)
plot_trace_plot(evaluator, burnin=10)



###############################################################################
# Compare Multiple Inference Methods
###############################################################################
init = sampler.prior_init()
sampler = LGSSMSampler(n=2, m=2, observations=data['observations'])

max_time = 60
## Fit Gibbs saving sample every second
gibbs_parameters, gibbs_time = sampler.fit_timed(
    iter_type='Gibbs',
    init_parameters=init, max_time=max_time, min_save_time=1, tqdm=tqdm,
    )

## Fit SGRLD (No Buffer)
nobuffer_parameters, nobuffer_time = sampler.fit_timed(
    iter_type='SGRLD',
    epsilon=0.1, subsequence_length=8, buffer_length=0,
    init_parameters=init, max_time=max_time, min_save_time=1, tqdm=tqdm,
    )

## Fit SGRLD (Buffer)
buffer_parameters, buffer_time = sampler.fit_timed(
    iter_type='SGRLD',
    epsilon=0.1, subsequence_length=8, buffer_length=4,
    init_parameters=init, max_time=max_time, min_save_time=1, tqdm=tqdm,
    )

## Evaluate
evaluators = {}
from sgmcmc_ssm.evaluator import half_average_parameters_list
evaluators['Gibbs'] = OfflineEvaluator(sampler,
         parameters_list=half_average_parameters_list(gibbs_parameters),
         parameters_times=gibbs_time,
         metric_functions = metric_functions,
         sample_functions = sample_functions,
         )
evaluators['SGRLD No Buffer'] = OfflineEvaluator(sampler,
         parameters_list=half_average_parameters_list(nobuffer_parameters),
         parameters_times=nobuffer_time,
         metric_functions = metric_functions,
         sample_functions = sample_functions,
         )
evaluators['SGRLD Buffer'] = OfflineEvaluator(sampler,
         parameters_list=half_average_parameters_list(buffer_parameters),
         parameters_times=buffer_time,
         metric_functions = metric_functions,
         sample_functions = sample_functions,
         )

for evaluator in tqdm(evaluators.values()):
    evaluator.evaluate(40, tqdm=tqdm)

# Plot Results
from sgmcmc_ssm.plotting_utils import compare_metrics
compare_metrics(evaluators, x='time')




