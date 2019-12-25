import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sgmcmc_ssm.models.gauss_hmm import GaussHMMSampler
from tqdm import tqdm

np.random.seed(12345)

# Load and Scale Data
from scipy.io import loadmat
ion_data = loadmat('data/alamethicin.mat')

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
observations = scaler.fit_transform(ion_data['originaldata'][1095:-3000])
filtered_observations = scaler.transform(ion_data['filtereddata'])
T = len(observations)

# Plot Data
fig, ax = plt.subplots(1,1)
ax.plot(np.arange(T)[::50], observations[::50], '-', label='scaled data')
ax.plot(np.arange(T)[::50], filtered_observations[::50], '-', label='scaled filtered data')
ax.set_title('Scaled Ion Data')
ax.set_xlabel('Time')
ax.set_ylabel('Voltage (Scaled)')
ax.legend()

# Process all
Y = observations[:-1000000]
filtered_Y = filtered_observations[:-1000000]
Y_test = observations[-1000000:]
filtered_Y_test = filtered_observations[-1000000:]


# Fit Gauss HMM using Gibbs Sampling -> Not Feasible -> 1 hour per iteration
#sampler = GaussHMMSampler(num_states=6, m=1, observations=Y)
#sampler.init_parameters_from_k_means(observations=Y, n_init=20, verbose=True)
#gibbs_parameters, gibbs_time = sampler.fit_timed(iter_type='Gibbs',
#        max_time=5*60,
#        tqdm=tqdm, tqdm_iter=True,
#        )


# Fit Gauss HMM using SGRLD
sampler = GaussHMMSampler(num_states=6, m=1, observations=Y)
sampler.init_parameters_from_k_means(observations=Y[::50], n_init=20, verbose=True)
sampler.parameters.pi_type = 'expanded'
sgrld_parameters, sgrld_time = sampler.fit_timed(
        iter_type='SGRLD',
        max_time=5*60,
        epsilon=0.001, subsequence_length=4, buffer_length=2,
        tqdm=tqdm,
        )

# Compare Fit at Init and Final On Subset
def compare_inference(parameters, sampler, Y, filtered_Y, tqdm=None):
    sampler.parameters = parameters.copy()
    z_prob = sampler.predict(
            observations=Y,
            target='latent', return_distr=True, tqdm=tqdm,
            )
    z_map = np.argmax(z_prob, axis=1)

    fig, axes = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios':[3,1]})
    axes[0].plot(Y, '-k', label='observations')
    for k in range(sampler.num_states):
        x = Y.copy()
        x[z_map != k] = np.nan
        axes[0].plot(x, '-C{}'.format(k), label='MAP(z) = {}'.format(k))
    axes[0].plot(filtered_Y, '--', color='gray', label='filtered observations')
    axes[0].legend()
    axes[1].plot(z_prob, '-')
    axes[1].set_ylabel('Latent Prob')
    return fig, axes

fig, axes = compare_inference(sgrld_parameters[0], sampler,
        Y_test, filtered_Y_test, tqdm=tqdm)
fig.suptitle('Init SGRLD Fit')
fig, axes = compare_inference(sgrld_parameters[-1], sampler,
        Y_test, filtered_Y_test, tqdm=tqdm)
fig.suptitle('Final SGRLD Fit')


################################################################################
# Sampler Evaluation
################################################################################
from sgmcmc_ssm.evaluator import OfflineEvaluator, half_average_parameters_list
from sgmcmc_ssm.metric_functions import (
        sample_function_parameters,
        noisy_logjoint_loglike_metric,
        noisy_predictive_logjoint_loglike_metric,
        )

# Evaluate Predictive Loglikelihood on Test Set
metric_functions=[
    noisy_predictive_logjoint_loglike_metric(
            num_steps_ahead=3, observations=Y_test, tqdm=tqdm),
    ]
sample_functions=sample_function_parameters(['pi', 'logit_pi', 'mu', 'R'])


# Evaluate SGRLD samples
sgrld_evaluator = OfflineEvaluator(sampler,
        parameters_list=half_average_parameters_list(sgrld_parameters),
        parameters_times=sgrld_time,
        metric_functions = metric_functions,
        sample_functions = sample_functions,
        )
sgrld_evaluator.evaluate(16, tqdm=tqdm)

# Plot Traces, Metrics, and Compare
from sgmcmc_ssm.plotting_utils import (
        plot_trace_plot,
        plot_metrics,
        compare_metrics,
        )

plot_trace_plot(sgrld_evaluator)
plot_metrics(sgrld_evaluator)

plot_metrics(sgrld_evaluator, x='time')

# EOF
