import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from sgmcmc_ssm.models.svm import SeqSVMSampler
from tqdm import tqdm

np.random.seed(12345)

###############################################################################
# Load and Scale Data
###############################################################################
exchange_data = np.load('data/EURUS_processed.npz')
print(list(exchange_data.keys()))
hourly_log_returns = exchange_data['hourly_log_returns']
hourly_dates = exchange_data['hourly_date']
print(hourly_log_returns.shape)

#from sklearn.preprocessing import RobustScaler
#scaler = RobustScaler()
#observations = scaler.fit_transform(hourly_log_returns.reshape(-1,1))
observations = hourly_log_returns.reshape(-1,1) * 1000
print(observations.shape)

###############################################################################
# Plot Data
###############################################################################
fig, ax = plt.subplots(1,1)
ax.plot(hourly_dates, observations)


###############################################################################
# Split Data on Gaps > 6 Hour
###############################################################################
gap_indices = np.where(np.diff(hourly_dates) > pd.Timedelta('6h'))[0].tolist()
split_observations = []
for start, end in zip([0]+gap_indices, gap_indices+[observations.size]):
    if end - start > 6:
        split_observations.append(observations[start:end])
split_summaries = pd.DataFrame([
    dict(max=np.max(obs), mean=np.mean(obs), min=np.min(obs), num=len(obs))
    for obs in split_observations])
print(np.around(split_summaries, decimals=2))

###############################################################################
# Fit SVM
###############################################################################
# Only Fit/Evaluate on the first 5 segments
sampler = SeqSVMSampler(n=1, m=1, observations=split_observations[0:5])
sampler.prior_init()
sampler.project_parameters()

print(sampler.noisy_logjoint(kind='pf', pf='paris', N=1000,
    return_loglike=True, tqdm=tqdm))

def compare_smoothed_pfs(list_of_kwargs):
    means_covs = {}
    for kwargs in list_of_kwargs:
        name = '{0} {1}'.format(kwargs.get('pf','Poyiadjis O(N)'),
                kwargs.get('N'))
        means_covs[name] = sampler.predict(target='latent', kind='pf',
                tqdm=tqdm, **kwargs)

    for jj, observation in enumerate(sampler.observations):
        fig, ax = plt.subplots(1, 1)
        for ii, (name, mean_cov) in enumerate(means_covs.items()):
            x_mean = mean_cov[jj][0][:,0]
            x_cov = mean_cov[jj][1][:,0, 0]
            ax.plot(x_mean, '-C{0}'.format(ii), label=name)
            ax.plot(x_mean+np.sqrt(x_cov), "--C{}".format(ii), alpha=0.5)
            ax.plot(x_mean-np.sqrt(x_cov), "--C{}".format(ii), alpha=0.5)
        ax.plot(np.log(observation**2)-np.log(sampler.parameters.R), '.k',
                label='log(data^2) - log(R)')
        ax.legend()
        ax.set_title('observations[{}]'.format(jj))
    return fig, ax

list_of_kwargs = [
        dict(N = 100),
        dict(N = 1000),
        dict(N = 10000),
        dict(pf='paris', N = 100),
        dict(pf='paris', N = 1000),
#        dict(pf='paris', N = 10000),
        ]
fig, ax = compare_smoothed_pfs(list_of_kwargs)


# Fit using SGLD
sgld_parameters, sgld_time = sampler.fit_timed(
        iter_type='SGLD',
        epsilon=0.001, subsequence_length=16, num_sequences=1, buffer_length=4,
        kind='pf', pf_kwargs=dict(pf='poyiadjis_N', N=1000),
        max_time=5*60,
        tqdm=tqdm, #tqdm_iter=True,
        )
print(sampler.noisy_logjoint(kind='pf', pf='paris', N=1000,
    return_loglike=True, tqdm=tqdm))

# Fit using LD (SGLD with S = T, for all sequences)
sampler.parameters = sgld_parameters[0].copy()
ld_parameters, ld_time = sampler.fit_timed(
        iter_type='SGLD',
        epsilon=0.1, subsequence_length=-1, num_sequences=-1, buffer_length=0,
        kind='pf', pf_kwargs=dict(pf='paris', N=1000),
        max_time=5*60,
        tqdm=tqdm, tqdm_iter=True,
        )
print(sampler.noisy_logjoint(kind='pf', pf='paris', N=1000,
    return_loglike=True, tqdm=tqdm))

###############################################################################
# Evaluate Fit
###############################################################################
from sgmcmc_ssm.evaluator import OfflineEvaluator, half_average_parameters_list
from sgmcmc_ssm.metric_functions import (
        sample_function_parameters,
        noisy_logjoint_loglike_metric,
        )
# Evaluate Loglikelihood on Training Set
metric_functions=[
    noisy_logjoint_loglike_metric(tqdm=tqdm, kind='pf', pf='paris', N=1000),
    ]
sample_functions=sample_function_parameters(['A', 'Q', 'R'])

# Evaluate SGLD samples
sgld_evaluator = OfflineEvaluator(sampler,
        parameters_list=sgld_parameters,
        parameters_times=sgld_time,
        metric_functions = metric_functions,
        sample_functions = sample_functions,
        )
sgld_evaluator.evaluate(40, tqdm=tqdm)

ld_evaluator = OfflineEvaluator(sampler,
        parameters_list=ld_parameters,
        parameters_times=ld_time,
        metric_functions = metric_functions,
        sample_functions = sample_functions,
        )
ld_evaluator.evaluate(40, tqdm=tqdm)


# Plot Traces, Metrics, and Compare
from sgmcmc_ssm.plotting_utils import (
        plot_trace_plot,
        plot_metrics,
        compare_metrics,
        )

plot_trace_plot(sgld_evaluator)
plot_metrics(sgld_evaluator)

plot_trace_plot(ld_evaluator)
plot_metrics(ld_evaluator)

compare_metrics(dict(
    SGLD=sgld_evaluator,
    LD=ld_evaluator,
    ),
    x='time',
    )
#


