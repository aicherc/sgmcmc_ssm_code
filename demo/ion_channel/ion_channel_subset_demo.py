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

# Plot Data
fig, ax = plt.subplots(1,1)
ax.plot(observations[-500000::50], '-', label='scaled data')
ax.plot(filtered_observations[-500000::50], '-', label='scaled filtered data')
ax.set_title('Scaled Downsampled Ion Data')
ax.set_xlabel('Time')
ax.set_ylabel('Voltage (Scaled)')
ax.legend()

# Only Process a subset for this example
Y = observations[-500000::50]
filtered_Y = filtered_observations[-500000::50]


# Fit Gauss HMM using Gibbs Sampling
sampler = GaussHMMSampler(num_states=6, m=1, observations=Y)
sampler.init_parameters_from_k_means(observations=Y[:10000], n_init=20)
gibbs_parameters, gibbs_time = sampler.fit_timed(iter_type='Gibbs',
        max_time=60,
        tqdm=tqdm,
        )

# Compare Fit at Init and Final
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

fig, axes = compare_inference(gibbs_parameters[0], sampler, Y, filtered_Y)
fig.suptitle('Init Gibbs Fit')
fig, axes = compare_inference(gibbs_parameters[-1], sampler, Y, filtered_Y)
fig.suptitle('Final Gibbs Fit')

# Fit Gauss HMM using SGRLD
sampler = GaussHMMSampler(num_states=6, m=1, observations=Y)
sampler.init_parameters_from_k_means(observations=Y[:10000], n_init=20)
sampler.parameters.pi_type = 'expanded'
sgrld_parameters, sgrld_time = sampler.fit_timed(
        iter_type='SGRLD',
        max_time=60,
        epsilon=0.001, subsequence_length=4, buffer_length=2,
        tqdm=tqdm,
        )


# Compare Fit at Init and Final
fig, axes = compare_inference(sgrld_parameters[0], sampler, Y, filtered_Y)
fig.suptitle('Init SGRLD Fit')
fig, axes = compare_inference(sgrld_parameters[-1], sampler, Y, filtered_Y)
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

# Evaluate Loglikelihood and Predictive Loglikelihood
metric_functions=[
    noisy_logjoint_loglike_metric(),
    noisy_predictive_logjoint_loglike_metric(num_steps_ahead=3),
    ]
sample_functions=sample_function_parameters(['pi', 'logit_pi', 'mu', 'R'])


# Evaluate Gibbs samples
gibbs_evaluator = OfflineEvaluator(sampler,
        parameters_list=half_average_parameters_list(gibbs_parameters),
        parameters_times=gibbs_time,
        metric_functions = metric_functions,
        sample_functions = sample_functions,
        )
gibbs_evaluator.evaluate(16, tqdm=tqdm)

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

plot_trace_plot(gibbs_evaluator)
plot_metrics(gibbs_evaluator)

plot_trace_plot(sgrld_evaluator)
plot_metrics(sgrld_evaluator)

compare_metrics(dict(
    Gibbs=gibbs_evaluator,
    SGRLD=sgrld_evaluator,
    ),
    x='time',
    )

# EOF
